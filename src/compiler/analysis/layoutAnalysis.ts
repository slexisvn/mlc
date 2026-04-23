// ─────────────────────────────────────────────────────────────────────────────
// Per-tensor layout fact inference and conflict detection.
//
// The analysis does a single forward pass over a topologically sorted graph
// and computes a TensorLayoutFact for every tensor by applying these rules:
//
//   1. Graph inputs: seeded from the tensor name (looks for "NCHW" / "NHWC"
//      substrings) or marked UNKNOWN when no annotation is found.
//   2. Layout-transforming ops (transpose, reshape): output layout taken
//      directly from the node's toLayout attr; falls back to UNKNOWN.
//   3. Layout-agnostic ops or unregistered ops: output layout propagated from
//      the primary (first) input.
//   4. Layout-preserving ops: output layout same as primary input.
//   5. Layout-sensitive ops: output layout taken from the contract's
//      outputLayout field; additionally, each input is checked against
//      requiredInputLayouts and a LayoutConflict is emitted on mismatch.
//
// Elimination candidates
// ──────────────────────
// After the forward pass, the analysis scans for consecutive transforming-op
// pairs whose transforms are inverses of each other (e.g. NCHW→NHWC followed
// immediately by NHWC→NCHW).  These are recorded as EliminationCandidates and
// can be acted on by LayoutTransformPass.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph } from "../ir/graph";
import {
  LayoutFormat, Layouts,
  LayoutTransform, getTransformFromAttrs, areInverseTransforms,
} from "../ir/layouts";
import { OpContractRegistry, DEFAULT_CONTRACT_REGISTRY } from "../ops/opContracts";
import { buildConsumerMap } from "../utils/graphUtils";
import { topoSort } from "../utils/toposort";

// ─── Public types ─────────────────────────────────────────────────────────────

export interface TensorLayoutFact {
  readonly tensorId:   string;
  readonly layout:     LayoutFormat;
  /** Confidence in the layout determination. */
  readonly confidence: "certain" | "inferred" | "unknown";
  /** How the layout was determined. */
  readonly source:     "annotation" | "propagated" | "default";
}

export interface LayoutConflict {
  readonly nodeId:         string;
  readonly op:             string;
  readonly inputTensorId:  string;
  readonly actualLayout:   LayoutFormat;
  readonly requiredLayout: string;    // may be a "|"-joined list
  readonly message:        string;
}

export interface EliminationCandidate {
  readonly firstNodeId:  string;
  readonly secondNodeId: string;
  readonly transform1:   LayoutTransform;
  readonly transform2:   LayoutTransform;
  readonly reason:       string;
}

export interface LayoutAnalysisResult {
  readonly tensorFacts:           Map<string, TensorLayoutFact>;
  readonly conflicts:             LayoutConflict[];
  readonly eliminationCandidates: EliminationCandidate[];
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function analyzeLayouts(
  graph:    Graph,
  registry: OpContractRegistry = DEFAULT_CONTRACT_REGISTRY,
): LayoutAnalysisResult {
  const tensorFacts            = new Map<string, TensorLayoutFact>();
  const conflicts:               LayoutConflict[]      = [];
  const eliminationCandidates:   EliminationCandidate[] = [];

  const { order } = topoSort(graph);
  const consumers  = buildConsumerMap(graph);

  // ── Seed graph inputs ────────────────────────────────────────────────────
  for (const tid of graph.inputIds) {
    const t          = graph.getTensor(tid);
    const annotated  = _detectLayoutFromName(t.name);
    tensorFacts.set(tid, {
      tensorId:   tid,
      layout:     annotated,
      confidence: annotated === Layouts.UNKNOWN ? "unknown" : "certain",
      source:     "annotation",
    });
  }

  // ── Forward pass ─────────────────────────────────────────────────────────
  for (const nid of order) {
    const node     = graph.getNode(nid);
    const contract = registry.get(node.op);

    // ── Layout-transforming ops (transpose, reshape) ───────────────────────
    if (registry.isLayoutTransforming(node.op)) {
      const xform = getTransformFromAttrs(node.attrs as Record<string, unknown>);
      for (const tid of node.outputs) {
        if (xform) {
          tensorFacts.set(tid, {
            tensorId:   tid,
            layout:     xform.toLayout,
            confidence: "certain",
            source:     "annotation",
          });
        } else {
          tensorFacts.set(tid, { tensorId: tid, layout: Layouts.UNKNOWN, confidence: "unknown", source: "default" });
        }
      }
      continue;
    }

    // ── Layout-agnostic or unregistered ops — propagate primary input ──────
    if (registry.isLayoutAgnostic(node.op) || contract === undefined) {
      const inputFact    = node.inputs[0] ? tensorFacts.get(node.inputs[0]) : undefined;
      const layout       = inputFact?.layout ?? Layouts.UNKNOWN;
      const confidence   = inputFact?.confidence === "certain" ? "inferred" as const : "unknown" as const;
      for (const tid of node.outputs) {
        tensorFacts.set(tid, { tensorId: tid, layout, confidence, source: "propagated" });
      }
      continue;
    }

    // ── Layout-preserving ops ──────────────────────────────────────────────
    if (registry.isLayoutPreserving(node.op)) {
      const inputFact  = node.inputs[0] ? tensorFacts.get(node.inputs[0]) : undefined;
      const layout     = inputFact?.layout ?? Layouts.UNKNOWN;
      const confidence = inputFact?.confidence === "certain" ? "inferred" as const : "unknown" as const;
      for (const tid of node.outputs) {
        tensorFacts.set(tid, { tensorId: tid, layout, confidence, source: "propagated" });
      }
      continue;
    }

    // ── Layout-sensitive ops ───────────────────────────────────────────────
    // Record output layout.
    const outLayout = contract.outputLayout ?? Layouts.UNKNOWN;
    const outConf   = contract.outputLayout ? "certain" as const : "unknown" as const;
    for (const tid of node.outputs) {
      tensorFacts.set(tid, { tensorId: tid, layout: outLayout, confidence: outConf, source: "annotation" });
    }

    // Check each input against required layouts.
    const required = contract.requiredInputLayouts ?? [];
    if (required.length > 0) {
      for (const tid of node.inputs) {
        const fact = tensorFacts.get(tid);
        if (!fact) continue;
        if (fact.layout === Layouts.UNKNOWN || fact.layout === Layouts.ANY) continue;
        if (!required.includes(fact.layout) && !required.includes(Layouts.ANY)) {
          conflicts.push({
            nodeId:         nid,
            op:             node.op,
            inputTensorId:  tid,
            actualLayout:   fact.layout,
            requiredLayout: required.join("|"),
            message:        `Op "${node.op}" requires layout [${required.join("|")}] ` +
                            `but tensor "${tid}" has layout "${fact.layout}".`,
          });
        }
      }
    }
  }

  // ── Elimination candidate detection ──────────────────────────────────────
  for (const nid of order) {
    const node = graph.getNode(nid);
    if (!registry.isLayoutTransforming(node.op)) continue;
    const xform1 = getTransformFromAttrs(node.attrs as Record<string, unknown>);
    if (!xform1 || node.outputs.length !== 1) continue;

    const outputTid = node.outputs[0];
    const cons      = consumers.get(outputTid) ?? [];
    if (cons.length !== 1) continue;   // branching: can't eliminate

    const nextNode = graph.getNode(cons[0]);
    if (!registry.isLayoutTransforming(nextNode.op)) continue;

    const xform2 = getTransformFromAttrs(nextNode.attrs as Record<string, unknown>);
    if (!xform2) continue;

    if (areInverseTransforms(xform1, xform2)) {
      eliminationCandidates.push({
        firstNodeId:  nid,
        secondNodeId: cons[0],
        transform1:   xform1,
        transform2:   xform2,
        reason:       `Consecutive inverse transforms: ` +
                      `${xform1.fromLayout}→${xform1.toLayout} ∘ ${xform2.fromLayout}→${xform2.toLayout} = identity`,
      });
    }
  }

  return { tensorFacts, conflicts, eliminationCandidates };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Infer a layout from a tensor's name by looking for well-known format strings.
 * Returns UNKNOWN when no known format is found.
 */
function _detectLayoutFromName(name: string): LayoutFormat {
  // Check in specificity order (longest / least-ambiguous first).
  for (const layout of [Layouts.NCHW, Layouts.NHWC, Layouts.NCW, Layouts.NWC, Layouts.NC]) {
    if (name.includes(layout)) return layout;
  }
  return Layouts.UNKNOWN;
}
