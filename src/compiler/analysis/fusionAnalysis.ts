// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic-grade fusion analysis.
//
// Improvement over running matchChains() directly
// ─────────────────────────────────────────────────
// matchChains() returns the set of approved chains but gives no insight into
// WHY other nodes were not fused.  This module wraps the matcher and, for
// every node NOT included in a confirmed chain, records the exact structural
// reason that prevented fusion.  The pass layer uses this information to emit
// per-candidate rejection logs instead of a single "No fusible chains found."
//
// Rejection reasons (RejectionReason)
// ─────────────────────────────────────
//   NoMatchingRule              — no registered rule starts with this op
//   MultiOutputIntermediateNode — an intermediate node exposes >1 output tensor
//   IntermediateTensorIsGraphOutput — a chain-internal tensor is also a graph output
//   BranchingIntermediateTensor — an intermediate tensor has >1 consumer
//   NodeAlreadyInChain          — node is already committed to another chain
//   OpContractForbidsFusion     — the op's contract marks it unfusible
//   PatternOpMismatch           — next op in chain does not match the rule pattern
// ─────────────────────────────────────────────────────────────────────────────

import { Graph } from "../ir/graph";
import { FusionRule, ChainCandidate } from "../ir/types";
import { OpContractRegistry, DEFAULT_CONTRACT_REGISTRY } from "../ops/opContracts";
import { matchChains } from "../patterns/matcher";
import { buildConsumerMap } from "../utils/graphUtils";
import { topoSort } from "../utils/toposort";

// ─── Public types ─────────────────────────────────────────────────────────────

export type RejectionReason =
  | "NoMatchingRule"
  | "MultiOutputIntermediateNode"
  | "IntermediateTensorIsGraphOutput"
  | "BranchingIntermediateTensor"
  | "NodeAlreadyInChain"
  | "OpContractForbidsFusion"
  | "PatternOpMismatch";

export interface RejectionRecord {
  readonly nodeId:         string;
  readonly op:             string;
  readonly reason:         RejectionReason;
  readonly detail:         string;
  readonly attemptedRule?: FusionRule;
}

/** A confirmed fusion candidate — superset of ChainCandidate with a mutable nodeIds array. */
export interface CandidateChain extends ChainCandidate {
  readonly rule:    FusionRule;
  readonly nodeIds: string[];
}

export interface FusionAnalysisResult {
  readonly candidates: CandidateChain[];
  readonly rejections: RejectionRecord[];
  readonly stats: {
    readonly totalNodes:        number;
    readonly fusibleCandidates: number;
    readonly rejectedNodes:     number;
  };
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Analyse the graph for fusion opportunities and return both the confirmed
 * candidates (from the matcher) and per-node rejection records for every node
 * that was NOT included in a chain.
 *
 * @param graph     The graph to analyse (not mutated).
 * @param rules     Fusion rules to evaluate (e.g. RuleRegistry.getRules()).
 * @param registry  Op contract registry; defaults to DEFAULT_CONTRACT_REGISTRY.
 */
export function analyzeFusion(
  graph:    Graph,
  rules:    readonly FusionRule[],
  registry: OpContractRegistry = DEFAULT_CONTRACT_REGISTRY,
): FusionAnalysisResult {
  const rejections: RejectionRecord[] = [];

  const { order, hasCycle } = topoSort(graph);
  if (hasCycle) {
    return {
      candidates: [],
      rejections: [{
        nodeId: "(graph)", op: "(graph)",
        reason: "NoMatchingRule",
        detail: "Graph contains a cycle — fusion analysis skipped.",
      }],
      stats: { totalNodes: graph.nodeOrder.length, fusibleCandidates: 0, rejectedNodes: graph.nodeOrder.length },
    };
  }

  const consumers      = buildConsumerMap(graph);
  const graphOutputSet = new Set(graph.outputIds);

  // Sort rules longest-first to mirror the matcher's tie-breaking logic.
  const sortedRules = [...rules]
    .filter(r => r.pattern.length >= 2)
    .sort((a, b) => b.pattern.length - a.pattern.length);

  // Run the authoritative matcher.
  const matchedChains  = matchChains(graph, rules);
  const committedNodes = new Set(matchedChains.flatMap(c => c.nodeIds));

  const candidates: CandidateChain[] = matchedChains.map(m => ({
    rule:    m.rule,
    nodeIds: [...m.nodeIds],
  }));

  // Diagnose every node that is NOT part of a confirmed chain.
  for (const nid of order) {
    if (committedNodes.has(nid)) continue;

    const node = graph.getNode(nid);

    // Check op-contract fusibility first.
    if (!registry.isFusible(node.op)) {
      rejections.push({
        nodeId: nid,
        op:     node.op,
        reason: "OpContractForbidsFusion",
        detail: `Op "${node.op}" is marked unfusible in the op contract registry.`,
      });
      continue;
    }

    // Try each rule to find the first structural rejection reason.
    let diagnosed = false;
    for (const rule of sortedRules) {
      if (node.op !== rule.pattern[0]) continue;

      const diag = _diagnoseChain(graph, nid, rule, consumers, graphOutputSet);
      if (diag !== null) {
        rejections.push({ nodeId: nid, op: node.op, ...diag, attemptedRule: rule });
        diagnosed = true;
        break;
      }
    }

    if (!diagnosed) {
      rejections.push({
        nodeId: nid,
        op:     node.op,
        reason: "NoMatchingRule",
        detail: `No fusion rule starts with op "${node.op}".`,
      });
    }
  }

  return {
    candidates,
    rejections,
    stats: {
      totalNodes:        graph.nodeOrder.length,
      fusibleCandidates: candidates.length,
      rejectedNodes:     rejections.length,
    },
  };
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Walk the chain starting at `startNodeId` according to `rule.pattern` and
 * return the first structural violation found, or null if all steps are valid
 * (meaning this chain should have been matched — useful for catching bugs).
 */
function _diagnoseChain(
  graph:          Graph,
  startNodeId:    string,
  rule:           FusionRule,
  consumers:      Map<string, string[]>,
  graphOutputSet: Set<string>,
): { reason: RejectionReason; detail: string } | null {
  let currentNodeId = startNodeId;

  for (let step = 1; step < rule.pattern.length; step++) {
    const currentNode = graph.getNode(currentNodeId);

    // Each intermediate node must emit exactly one output.
    if (currentNode.outputs.length !== 1) {
      return {
        reason: "MultiOutputIntermediateNode",
        detail: `Node "${currentNodeId}" (${currentNode.op}) has ` +
                `${currentNode.outputs.length} output(s); fusion requires exactly 1.`,
      };
    }

    const tid = currentNode.outputs[0];

    // Intermediate tensors must not be graph outputs.
    if (graphOutputSet.has(tid)) {
      return {
        reason: "IntermediateTensorIsGraphOutput",
        detail: `Tensor "${tid}" produced by "${currentNodeId}" (${currentNode.op}) ` +
                `is a graph output and cannot be an internal chain edge.`,
      };
    }

    // Intermediate tensors must have exactly one consumer.
    const cons = consumers.get(tid) ?? [];
    if (cons.length !== 1) {
      return {
        reason: "BranchingIntermediateTensor",
        detail: `Tensor "${tid}" has ${cons.length} consumer(s): ` +
                `[${cons.join(", ")}]. A fusible chain requires a single-consumer path.`,
      };
    }

    const nextNodeId = cons[0];
    const nextNode   = graph.getNode(nextNodeId);

    // Next op must match the pattern.
    if (nextNode.op !== rule.pattern[step]) {
      return {
        reason: "PatternOpMismatch",
        detail: `Expected op "${rule.pattern[step]}" at step ${step} ` +
                `(rule "${rule.name ?? rule.pattern.join("→")}") ` +
                `but found "${nextNode.op}" (node "${nextNodeId}").`,
      };
    }

    currentNodeId = nextNodeId;
  }

  // All checks passed — chain should have matched.
  return null;
}
