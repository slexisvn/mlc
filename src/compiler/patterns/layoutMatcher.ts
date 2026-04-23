// ─────────────────────────────────────────────────────────────────────────────
//
// Layout chain matcher — mirrors the fusion matcher's greedy algorithm but
// operates on LayoutRewriteRules instead of FusionRules.
//
// Algorithm
// ─────────
// 1. Topologically sort the graph.
// 2. Sort rules by priority (highest first) so more-specific rules win.
// 3. For each unvisited node, attempt each rule in priority order:
//      a. The node's op must match rule.pattern[0].
//      b. Walk forward: each intermediate node must have exactly one output
//         tensor and exactly one consumer; the consumer must match the next
//         pattern op.
//      c. On a successful match, mark all chain nodes as used and collect
//         any LayoutTransform descriptors from transforming ops.
// 4. Return all matched layout chains.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph } from "../ir/graph";
import { LayoutTransform, getTransformFromAttrs } from "../ir/layouts";
import { OpContractRegistry, DEFAULT_CONTRACT_REGISTRY } from "../ops/opContracts";
import { LayoutRewriteRule } from "./layoutRules";
import { buildConsumerMap } from "../utils/graphUtils";
import { topoSort } from "../utils/toposort";

// ─── Public types ─────────────────────────────────────────────────────────────

export interface MatchedLayoutChain {
  readonly rule:       LayoutRewriteRule;
  /** Node ids in chain order (first → last). */
  readonly nodeIds:    string[];
  /**
   * Layout transforms collected from transforming ops in the chain.
   * For a cancellation rule the first and last transforms should be inverses.
   */
  readonly transforms: LayoutTransform[];
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Find all non-overlapping layout chains in the graph that match a registered
 * LayoutRewriteRule.
 *
 * @param graph     Graph to inspect (not mutated).
 * @param rules     Ordered set of layout rules (from LayoutRuleRegistry.getRules()).
 * @param registry  Op contract registry; defaults to DEFAULT_CONTRACT_REGISTRY.
 */
export function matchLayoutChains(
  graph:    Graph,
  rules:    readonly LayoutRewriteRule[],
  registry: OpContractRegistry = DEFAULT_CONTRACT_REGISTRY,
): MatchedLayoutChain[] {
  const { order, hasCycle } = topoSort(graph);
  if (hasCycle) return [];

  const consumers      = buildConsumerMap(graph);
  const graphOutputSet = new Set(graph.outputIds);
  // Sort rules highest-priority first.
  const sortedRules    = [...rules].sort((a, b) => b.priority - a.priority);
  const used           = new Set<string>();
  const matches: MatchedLayoutChain[] = [];

  for (const startNodeId of order) {
    if (used.has(startNodeId)) continue;

    for (const rule of sortedRules) {
      const match = _tryMatchLayoutChain(
        graph, rule, startNodeId, consumers, graphOutputSet, used, registry,
      );
      if (match !== null) {
        for (const nid of match.nodeIds) used.add(nid);
        matches.push(match);
        break;   // only one rule per start node
      }
    }
  }

  return matches;
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

function _tryMatchLayoutChain(
  graph:          Graph,
  rule:           LayoutRewriteRule,
  startNodeId:    string,
  consumers:      Map<string, string[]>,
  graphOutputSet: Set<string>,
  used:           Set<string>,
  registry:       OpContractRegistry,
): MatchedLayoutChain | null {
  const startNode = graph.getNode(startNodeId);
  if (startNode.op !== rule.pattern[0]) return null;

  const chain:      string[]          = [startNodeId];
  const transforms: LayoutTransform[] = [];
  let   currentNodeId                  = startNodeId;

  // Collect transform from the start node if it is a transforming op.
  _collectTransform(graph.getNode(startNodeId), registry, transforms);

  for (let step = 1; step < rule.pattern.length; step++) {
    const currentNode = graph.getNode(currentNodeId);
    if (currentNode.outputs.length !== 1) return null;

    const outputTid  = currentNode.outputs[0];
    // Allow the last node's output to be a graph output (it survives the rewrite).
    // Intermediate outputs must not be graph outputs.
    const isLastStep = step === rule.pattern.length - 1;
    if (!isLastStep && graphOutputSet.has(outputTid)) return null;

    const cons = consumers.get(outputTid) ?? [];
    if (cons.length !== 1) return null;

    const nextNodeId = cons[0];
    if (used.has(nextNodeId)) return null;

    const nextNode = graph.getNode(nextNodeId);
    if (nextNode.op !== rule.pattern[step]) return null;

    chain.push(nextNodeId);
    _collectTransform(nextNode, registry, transforms);
    currentNodeId = nextNodeId;
  }

  return { rule, nodeIds: chain, transforms };
}

function _collectTransform(
  node:      import("../ir/graph").Node,
  registry:  OpContractRegistry,
  out:       LayoutTransform[],
): void {
  if (!registry.isLayoutTransforming(node.op)) return;
  const t = getTransformFromAttrs(node.attrs as Record<string, unknown>);
  if (t) out.push(t);
}
