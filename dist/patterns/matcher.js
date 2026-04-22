"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// patterns/matcher.ts
//
// Rule-based linear chain matcher.
//
// Algorithm
// ─────────
// 1. Topologically sort the graph for a deterministic traversal order.
// 2. Sort candidate rules longest-first so a 3-op rule beats any 2-op prefix.
// 3. For each unvisited node (in topo order), attempt every rule:
//      a. The start node's op must match rule.pattern[0].
//      b. Walk forward one step at a time:
//           • The current node must have exactly ONE output tensor.
//           • That tensor must NOT be a graph output (it's an internal edge).
//           • That tensor must have exactly ONE consumer (no branching).
//           • The consumer's op must match the next pattern element.
//      c. On success, mark all chain nodes as "used" (no overlap between chains).
// 4. Return all matched chains as MatchedChain objects.
//
// Safety note:
//   The matcher only discovers chains; it does NOT modify the graph.
//   All graph mutations are delegated to the FusionPass.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.matchChains = matchChains;
const graphUtils_1 = require("../utils/graphUtils");
const toposort_1 = require("../utils/toposort");
// ─── Matcher entry-point ──────────────────────────────────────────────────────
/**
 * Find all non-overlapping fusible chains in the graph using the given rules.
 *
 * Returns an ordered list of MatchedChain objects ready for the FusionPass to
 * process.  No chain shares a node with another (greedy, first-match wins per
 * start node).
 */
function matchChains(graph, rules) {
    const { order, hasCycle } = (0, toposort_1.topoSort)(graph);
    if (hasCycle)
        return []; // Guard: cannot match reliably in a cyclic graph.
    const consumers = (0, graphUtils_1.buildConsumerMap)(graph);
    const graphOutputSet = new Set(graph.outputIds);
    // Sort rules longest-first for greedy best-match.
    const sortedRules = [...rules]
        .filter(r => r.pattern.length >= 2)
        .sort((a, b) => b.pattern.length - a.pattern.length);
    const used = new Set();
    const matches = [];
    for (const startNodeId of order) {
        if (used.has(startNodeId))
            continue;
        for (const rule of sortedRules) {
            const chain = tryMatchChain(graph, rule, startNodeId, consumers, graphOutputSet, used);
            if (chain !== null) {
                // Commit all nodes of this chain so no other rule can overlap them.
                for (const nid of chain.nodeIds)
                    used.add(nid);
                matches.push(chain);
                break; // Move to the next start node.
            }
        }
    }
    return matches;
}
// ─── Internal chain-walk ──────────────────────────────────────────────────────
/**
 * Attempt to grow a chain starting at `startNodeId` that satisfies `rule`.
 * Returns null if any step of the pattern is not satisfied.
 */
function tryMatchChain(graph, rule, startNodeId, consumers, graphOutputSet, used) {
    const startNode = graph.getNode(startNodeId);
    // Step 0: the start node must match the first pattern element.
    if (startNode.op !== rule.pattern[0])
        return null;
    const chain = [startNodeId];
    let currentNodeId = startNodeId;
    // Steps 1..N-1: extend the chain one node at a time.
    for (let step = 1; step < rule.pattern.length; step++) {
        const currentNode = graph.getNode(currentNodeId);
        // ① The current node must have exactly one output tensor so the chain
        //   is unambiguous (multi-output nodes can't be safely fused mid-chain
        //   because one of their outputs might escape the chain).
        if (currentNode.outputs.length !== 1)
            return null;
        const outputTid = currentNode.outputs[0];
        // ② The intermediate tensor must NOT be a graph output; we cannot
        //   "cut through" an observable value.
        if (graphOutputSet.has(outputTid))
            return null;
        // ③ The intermediate tensor must have exactly one consumer — no branching.
        const cons = consumers.get(outputTid) ?? [];
        if (cons.length !== 1)
            return null;
        const nextNodeId = cons[0];
        // ④ The next node must not already belong to another committed chain.
        if (used.has(nextNodeId))
            return null;
        // ⑤ The next node's op must match the expected pattern element.
        const nextNode = graph.getNode(nextNodeId);
        if (nextNode.op !== rule.pattern[step])
            return null;
        chain.push(nextNodeId);
        currentNodeId = nextNodeId;
    }
    return { rule, nodeIds: chain };
}
//# sourceMappingURL=matcher.js.map