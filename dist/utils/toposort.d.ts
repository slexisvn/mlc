import { Graph } from "../ir/graph";
export interface TopoSortResult {
    /** Node ids in topological (dependency-first) order. */
    order: string[];
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
export declare function topoSort(graph: Graph): TopoSortResult;
