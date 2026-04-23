// ─────────────────────────────────────────────────────────────────────────────
//
// Layout-aware graph rewrite pass.
//
// Pipeline (two strict phases — no interleaving):
//   Phase 1 — Analysis:
//     • Run analyzeLayouts() to build per-tensor layout facts.
//     • Emit warnings for any detected LayoutConflicts (op requires a layout
//       that its input does not have).
//     • Log the number of elimination candidates found.
//
//   Phase 2 — Pattern matching + rewriting:
//     • Run matchLayoutChains() to find chains matching LayoutRewriteRules.
//     • For each match, dispatch to the appropriate rewrite handler:
//         "cancellation" → _applyCancellation: remove back-to-back inverse transposes.
//         "propagation"  → _applyPropagation:  remove sandwich transposes around
//                                               layout-agnostic middle ops.
//     • Return the rewritten graph (or the original if nothing changed).
//
// Safety guarantees
// ─────────────────
// • The input graph is cloned before any mutation.
// • Every rewrite checks preconditions before committing changes.
// • If any precondition fails the chain is skipped (logged as "skipped").
// • Structural invariants are maintained: SSA, single-producer per tensor,
//   no dangling edges.  The PassManager validates after each pass.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph } from "../ir/graph";
import { LayoutTransform } from "../ir/layouts";
import { Pass, PassLog, PassResult } from "./pass";
import { LayoutRuleRegistry } from "../patterns/layoutRules";
import { matchLayoutChains, MatchedLayoutChain } from "../patterns/layoutMatcher";
import { analyzeLayouts } from "../analysis/layoutAnalysis";
import { OpContractRegistry, DEFAULT_CONTRACT_REGISTRY } from "../ops/opContracts";
import { transformsCancel } from "../optimizer/layoutCanonicalizer";
import { buildConsumerMap } from "../utils/graphUtils";

export class LayoutTransformPass implements Pass {
  readonly name = "LayoutTransformPass";

  constructor(
    private readonly ruleRegistry: LayoutRuleRegistry,
    private readonly opRegistry:   OpContractRegistry = DEFAULT_CONTRACT_REGISTRY,
  ) {}

  run(graph: Graph): PassResult {
    const logs:      PassLog[] = [];
    const workGraph            = graph.clone();

    // ── Phase 1: layout analysis ───────────────────────────────────────────
    const layoutResult = analyzeLayouts(workGraph, this.opRegistry);

    for (const conflict of layoutResult.conflicts) {
      logs.push({ level: "warn", message: `Layout conflict: ${conflict.message}` });
    }

    logs.push({
      level:   "info",
      message: `Layout analysis: ${layoutResult.eliminationCandidates.length} elimination ` +
               `candidate(s), ${layoutResult.conflicts.length} conflict(s).`,
    });

    // ── Phase 2: match layout chains ──────────────────────────────────────
    const matches = matchLayoutChains(
      workGraph, this.ruleRegistry.getRules(), this.opRegistry,
    );

    if (matches.length === 0) {
      logs.push({ level: "info", message: "No layout chains matched — graph unchanged." });
      return { graph, changed: false, logs };
    }

    logs.push({ level: "info", message: `Found ${matches.length} layout chain(s) to process.` });

    let changed = false;
    for (const match of matches) {
      if (this._applyLayoutRewrite(workGraph, match, logs)) changed = true;
    }

    return changed
      ? { graph: workGraph, changed: true, logs }
      : { graph, changed: false, logs };
  }

  // ─── Dispatch ─────────────────────────────────────────────────────────────

  private _applyLayoutRewrite(
    graph: Graph,
    match: MatchedLayoutChain,
    logs:  PassLog[],
  ): boolean {
    const { rule, nodeIds, transforms } = match;
    const chainDesc = nodeIds
      .map(id => `${id}(${graph.getNode(id).op})`)
      .join(" → ");

    logs.push({ level: "info", message: `Processing [${rule.name}]: [${chainDesc}]` });

    if (rule.kind === "cancellation") {
      return this._applyCancellation(graph, nodeIds, transforms, logs);
    }
    if (rule.kind === "propagation") {
      return this._applyPropagation(graph, nodeIds, transforms, logs);
    }

    logs.push({ level: "warn", message: `  Unknown rule kind "${rule.kind}" — skipping.` });
    return false;
  }

  // ─── Cancellation rewrite ─────────────────────────────────────────────────
  //
  // Removes two consecutive transpose nodes whose transforms compose to
  // the identity.
  //
  //   inputTid → [transpose A] → intermediateTid → [transpose B] → outputTid
  //
  // After the rewrite, all consumers of `outputTid` reference `inputTid`
  // directly.  Both transpose nodes and both intermediate tensors are removed.

  private _applyCancellation(
    graph:      Graph,
    nodeIds:    string[],
    transforms: readonly LayoutTransform[],
    logs:       PassLog[],
  ): boolean {
    if (nodeIds.length !== 2) {
      logs.push({ level: "info", message: "  Cancellation requires exactly 2 nodes — skipping." });
      return false;
    }
    if (transforms.length < 2) {
      logs.push({ level: "info", message: "  Could not extract transform descriptors — skipping." });
      return false;
    }
    if (!transformsCancel(transforms[0], transforms[1])) {
      logs.push({ level: "info", message: "  Transforms are not inverses — skipping." });
      return false;
    }

    const firstNode  = graph.getNode(nodeIds[0]);
    const secondNode = graph.getNode(nodeIds[1]);

    if (firstNode.inputs.length !== 1 || secondNode.outputs.length !== 1) {
      logs.push({ level: "info", message: "  Unexpected multi-input/output structure — skipping." });
      return false;
    }

    const inputTid        = firstNode.inputs[0];
    const intermediateTid = firstNode.outputs[0];
    const outputTid       = secondNode.outputs[0];

    // The intermediate tensor must be exclusively consumed by the second transpose.
    const consumers = buildConsumerMap(graph);
    const intCons   = consumers.get(intermediateTid) ?? [];
    if (intCons.length !== 1) {
      logs.push({
        level:   "info",
        message: `  Intermediate tensor "${intermediateTid}" has ${intCons.length} consumer(s) — cannot eliminate.`,
      });
      return false;
    }

    logs.push({
      level:   "info",
      message: `  Cancelling ${nodeIds[0]} + ${nodeIds[1]}: ` +
               `rewiring consumers of "${outputTid}" → "${inputTid}"`,
    });

    // Rewire all node inputs that reference outputTid → inputTid.
    for (const node of [...graph.nodes.values()]) {
      const updated = [...node.inputs].map(t => (t === outputTid ? inputTid : t));
      if (updated.some((t, i) => t !== node.inputs[i])) {
        graph._replaceNode(node.id, { ...node, inputs: updated });
      }
    }

    // Update graph output declarations.
    graph._replaceOutputTensor(outputTid, inputTid);

    // Remove both transpose nodes.
    graph._removeNode(nodeIds[0]);
    graph._removeNode(nodeIds[1]);

    // Remove the tensors they produced.
    graph._removeTensor(intermediateTid);
    graph._removeTensor(outputTid);

    return true;
  }

  // ─── Propagation rewrite ──────────────────────────────────────────────────
  //
  // Removes two "sandwich" transposes around a layout-agnostic middle op when
  // the outer transposes are inverses of each other.
  //
  //   inputTid → [transpose A] → edgeTid → [middle] → midOutTid → [transpose B] → outputTid
  //
  // After the rewrite:
  //   inputTid → [middle] → midOutTid  (middle's input rewired to inputTid)
  //
  // Consumers of `outputTid` now reference `midOutTid` directly.  Both
  // transpose nodes and their output tensors (edgeTid, outputTid) are removed.

  private _applyPropagation(
    graph:      Graph,
    nodeIds:    string[],
    transforms: readonly LayoutTransform[],
    logs:       PassLog[],
  ): boolean {
    if (nodeIds.length !== 3) {
      logs.push({ level: "info", message: "  Propagation requires exactly 3 nodes — skipping." });
      return false;
    }
    if (transforms.length < 2) {
      logs.push({ level: "info", message: "  Could not extract transform descriptors — skipping." });
      return false;
    }

    const firstTransform = transforms[0];
    const lastTransform  = transforms[transforms.length - 1];

    if (!transformsCancel(firstTransform, lastTransform)) {
      logs.push({ level: "info", message: "  Outer transposes are not inverses — skipping." });
      return false;
    }

    const firstTransposeNode  = graph.getNode(nodeIds[0]);
    const middleNode          = graph.getNode(nodeIds[1]);
    const secondTransposeNode = graph.getNode(nodeIds[2]);

    const inputTid   = firstTransposeNode.inputs[0];   // original pre-chain tensor
    const edgeTid    = firstTransposeNode.outputs[0];  // first-transpose → middle
    const midOutTid  = secondTransposeNode.inputs[0];  // middle → second-transpose
    const outputTid  = secondTransposeNode.outputs[0]; // post-chain tensor

    // Safety: the middle node must receive the chain edge as an input and
    // must produce the tensor that feeds the second transpose.
    if (!middleNode.inputs.includes(edgeTid)) {
      logs.push({ level: "info", message: `  Middle node "${nodeIds[1]}" does not consume chain edge — skipping.` });
      return false;
    }
    if (!middleNode.outputs.includes(midOutTid)) {
      logs.push({ level: "info", message: `  Middle node "${nodeIds[1]}" does not produce expected tensor — skipping.` });
      return false;
    }

    // Each intermediate tensor must be single-consumer.
    const consumers = buildConsumerMap(graph);
    if ((consumers.get(edgeTid)   ?? []).length !== 1 ||
        (consumers.get(midOutTid) ?? []).length !== 1) {
      logs.push({ level: "info", message: "  Intermediate tensor has multiple consumers — skipping." });
      return false;
    }

    logs.push({
      level:   "info",
      message: `  Propagating through ${nodeIds[1]}(${middleNode.op}): ` +
               `removing ${nodeIds[0]} and ${nodeIds[2]}`,
    });

    // Rewire the middle node: replace edgeTid → inputTid in its inputs.
    const updatedInputs = [...middleNode.inputs].map(t => (t === edgeTid ? inputTid : t));
    graph._replaceNode(middleNode.id, { ...middleNode, inputs: updatedInputs });

    // Rewire all node inputs that reference outputTid → midOutTid.
    for (const node of [...graph.nodes.values()]) {
      const updated = [...node.inputs].map(t => (t === outputTid ? midOutTid : t));
      if (updated.some((t, i) => t !== node.inputs[i])) {
        graph._replaceNode(node.id, { ...node, inputs: updated });
      }
    }

    // Update graph output declarations.
    graph._replaceOutputTensor(outputTid, midOutTid);

    // Remove both transpose nodes.
    graph._removeNode(nodeIds[0]);
    graph._removeNode(nodeIds[2]);

    // Remove the tensors they produced.
    graph._removeTensor(edgeTid);
    graph._removeTensor(outputTid);

    return true;
  }
}
