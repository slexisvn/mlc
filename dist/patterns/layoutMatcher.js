"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// patterns/layoutMatcher.ts
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.matchLayoutChains = matchLayoutChains;
const layouts_1 = require("../ir/layouts");
const opContracts_1 = require("../ops/opContracts");
const graphUtils_1 = require("../utils/graphUtils");
const toposort_1 = require("../utils/toposort");
// ─── Public API ───────────────────────────────────────────────────────────────
/**
 * Find all non-overlapping layout chains in the graph that match a registered
 * LayoutRewriteRule.
 *
 * @param graph     Graph to inspect (not mutated).
 * @param rules     Ordered set of layout rules (from LayoutRuleRegistry.getRules()).
 * @param registry  Op contract registry; defaults to DEFAULT_CONTRACT_REGISTRY.
 */
function matchLayoutChains(graph, rules, registry = opContracts_1.DEFAULT_CONTRACT_REGISTRY) {
    const { order, hasCycle } = (0, toposort_1.topoSort)(graph);
    if (hasCycle)
        return [];
    const consumers = (0, graphUtils_1.buildConsumerMap)(graph);
    const graphOutputSet = new Set(graph.outputIds);
    // Sort rules highest-priority first.
    const sortedRules = [...rules].sort((a, b) => b.priority - a.priority);
    const used = new Set();
    const matches = [];
    for (const startNodeId of order) {
        if (used.has(startNodeId))
            continue;
        for (const rule of sortedRules) {
            const match = _tryMatchLayoutChain(graph, rule, startNodeId, consumers, graphOutputSet, used, registry);
            if (match !== null) {
                for (const nid of match.nodeIds)
                    used.add(nid);
                matches.push(match);
                break; // only one rule per start node
            }
        }
    }
    return matches;
}
// ─── Internal helpers ─────────────────────────────────────────────────────────
function _tryMatchLayoutChain(graph, rule, startNodeId, consumers, graphOutputSet, used, registry) {
    const startNode = graph.getNode(startNodeId);
    if (startNode.op !== rule.pattern[0])
        return null;
    const chain = [startNodeId];
    const transforms = [];
    let currentNodeId = startNodeId;
    // Collect transform from the start node if it is a transforming op.
    _collectTransform(graph.getNode(startNodeId), registry, transforms);
    for (let step = 1; step < rule.pattern.length; step++) {
        const currentNode = graph.getNode(currentNodeId);
        if (currentNode.outputs.length !== 1)
            return null;
        const outputTid = currentNode.outputs[0];
        // Allow the last node's output to be a graph output (it survives the rewrite).
        // Intermediate outputs must not be graph outputs.
        const isLastStep = step === rule.pattern.length - 1;
        if (!isLastStep && graphOutputSet.has(outputTid))
            return null;
        const cons = consumers.get(outputTid) ?? [];
        if (cons.length !== 1)
            return null;
        const nextNodeId = cons[0];
        if (used.has(nextNodeId))
            return null;
        const nextNode = graph.getNode(nextNodeId);
        if (nextNode.op !== rule.pattern[step])
            return null;
        chain.push(nextNodeId);
        _collectTransform(nextNode, registry, transforms);
        currentNodeId = nextNodeId;
    }
    return { rule, nodeIds: chain, transforms };
}
function _collectTransform(node, registry, out) {
    if (!registry.isLayoutTransforming(node.op))
        return;
    const t = (0, layouts_1.getTransformFromAttrs)(node.attrs);
    if (t)
        out.push(t);
}
//# sourceMappingURL=layoutMatcher.js.map