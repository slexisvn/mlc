"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// utils/toposort.ts
//
// Topological sort (Kahn's BFS algorithm) over the node graph.
//
// Returns nodes in dependency-first order so passes and the execution planner
// can traverse operations safely.
//
// Design note:
//   We break ties alphabetically for a fully deterministic output, which makes
//   debugging and snapshot-testing reproducible.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.topoSort = topoSort;
/**
 * Compute a topological ordering of all nodes in the graph.
 *
 * Uses Kahn's BFS algorithm:
 *   1. Compute in-degree of every node.
 *   2. Seed a queue with all zero-in-degree nodes (sorted for determinism).
 *   3. Repeatedly dequeue a node, emit it, and decrement neighbors' in-degrees.
 *   4. If the emitted count < total count, a cycle exists.
 *
 * Multi-tensor edges between the same pair of nodes are deduplicated so
 * in-degrees reflect unique data-flow dependencies, not tensor multiplicity.
 */
function topoSort(graph) {
    const nodeIds = [...graph.nodeOrder];
    // Initialise per-node state.
    const inDegree = new Map();
    const adjOut = new Map();
    for (const nid of nodeIds) {
        inDegree.set(nid, 0);
        adjOut.set(nid, []);
    }
    // Build tensor → producer index.
    const tensorProducer = new Map();
    for (const node of graph.nodes.values()) {
        for (const tid of node.outputs)
            tensorProducer.set(tid, node.id);
    }
    // Collect unique directed edges.
    const seenEdges = new Set();
    for (const node of graph.nodes.values()) {
        for (const tid of node.inputs) {
            const prod = tensorProducer.get(tid);
            if (prod && prod !== node.id) {
                const key = `${prod}>>>${node.id}`;
                if (!seenEdges.has(key)) {
                    seenEdges.add(key);
                    adjOut.get(prod).push(node.id);
                    inDegree.set(node.id, (inDegree.get(node.id) ?? 0) + 1);
                }
            }
        }
    }
    // Seed queue with zero-in-degree nodes, sorted for determinism.
    const queue = [];
    for (const [nid, deg] of inDegree) {
        if (deg === 0)
            queue.push(nid);
    }
    queue.sort();
    const order = [];
    while (queue.length > 0) {
        queue.sort(); // keep deterministic after each expansion
        const cur = queue.shift();
        order.push(cur);
        for (const next of (adjOut.get(cur) ?? [])) {
            const newDeg = (inDegree.get(next) ?? 1) - 1;
            inDegree.set(next, newDeg);
            if (newDeg === 0)
                queue.push(next);
        }
    }
    return {
        order,
        hasCycle: order.length < nodeIds.length,
    };
}
//# sourceMappingURL=toposort.js.map