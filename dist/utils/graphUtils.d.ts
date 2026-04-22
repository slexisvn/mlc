import { Graph, Node, Tensor } from "../ir/graph";
export declare function resetFusedCounter(): void;
/**
 * Build a map from tensor id → list of node ids that consume it.
 *
 * Every tensor in the graph receives an entry (possibly with an empty list)
 * so callers can safely use `consumers.get(tid) ?? []` without undefined checks.
 */
export declare function buildConsumerMap(graph: Graph): Map<string, string[]>;
export interface ChainBoundary {
    /** Tensor ids consumed by the chain but produced outside it (or graph inputs). */
    externalInputs: string[];
    /** Tensor ids produced by the chain and observed outside it (or are graph outputs). */
    externalOutputs: string[];
    /** Tensor ids produced and consumed only inside the chain — safe to remove after fusion. */
    internalTensors: string[];
}
/**
 * Given a candidate chain of node ids, classify every tensor produced by those
 * nodes as either an external output or an internal (intermediate) tensor.
 *
 * A tensor is an external output when:
 *   - At least one of its consumers is a node outside the chain, OR
 *   - It is declared as a graph-level output.
 *
 * A tensor is an external input when:
 *   - Its producer is outside the chain (including graph-input tensors).
 *
 * The passed `consumers` map must already reflect the current graph state
 * (call buildConsumerMap immediately before this function when the graph may
 * have been mutated).
 */
export declare function extractChainBoundary(graph: Graph, chain: string[], consumers: Map<string, string[]>): ChainBoundary;
export interface FusedNodeResult {
    node: Node;
    tensors: Tensor[];
}
/**
 * Create a replacement Node for a fused chain and the new output Tensors it produces.
 *
 * The fused node:
 *   - Has the same external inputs as the chain (preserving data flow from outside).
 *   - Produces new tensors that mirror the chain's external outputs (same dtype/shape).
 *   - Carries a "fusedFrom" attribute listing the original output tensor ids for traceability.
 *
 * Callers must subsequently:
 *   1. Insert the node+tensors into the graph.
 *   2. Rewire all downstream consumers from old output ids to the new ids.
 *   3. Remove the original nodes and their internal / old-external tensors.
 */
export declare function createFusedNode(fusedOp: string, externalInputs: string[], externalOutputs: string[], graph: Graph): FusedNodeResult;
