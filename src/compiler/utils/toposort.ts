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

import { Graph } from "../ir/graph";

export interface TopoSortResult {
  /** Node ids in topological (dependency-first) order. */
  order:    string[];
  /** True when the graph contains a cycle and the sort is incomplete. */
  hasCycle: boolean;
}

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
export function topoSort(graph: Graph): TopoSortResult {
  const nodeIds = [...graph.nodeOrder];

  // Initialise per-node state.
  const inDegree = new Map<string, number>();
  const adjOut   = new Map<string, string[]>();
  for (const nid of nodeIds) {
    inDegree.set(nid, 0);
    adjOut.set(nid, []);
  }

  // Build tensor → producer index.
  const tensorProducer = new Map<string, string>();
  for (const node of graph.nodes.values()) {
    for (const tid of node.outputs) tensorProducer.set(tid, node.id);
  }

  // Collect unique directed edges.
  const seenEdges = new Set<string>();
  for (const node of graph.nodes.values()) {
    for (const tid of node.inputs) {
      const prod = tensorProducer.get(tid);
      if (prod && prod !== node.id) {
        const key = `${prod}>>>${node.id}`;
        if (!seenEdges.has(key)) {
          seenEdges.add(key);
          adjOut.get(prod)!.push(node.id);
          inDegree.set(node.id, (inDegree.get(node.id) ?? 0) + 1);
        }
      }
    }
  }

  // Seed queue with zero-in-degree nodes, sorted for determinism.
  const queue: string[] = [];
  for (const [nid, deg] of inDegree) {
    if (deg === 0) queue.push(nid);
  }
  queue.sort();

  const order: string[] = [];
  while (queue.length > 0) {
    queue.sort(); // keep deterministic after each expansion
    const cur = queue.shift()!;
    order.push(cur);
    for (const next of (adjOut.get(cur) ?? [])) {
      const newDeg = (inDegree.get(next) ?? 1) - 1;
      inDegree.set(next, newDeg);
      if (newDeg === 0) queue.push(next);
    }
  }

  return {
    order,
    hasCycle: order.length < nodeIds.length,
  };
}
