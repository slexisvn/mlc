// ─────────────────────────────────────────────────────────────────────────────
// utils/graphUtils.ts
//
// Shared graph manipulation helpers used by the pattern matcher, cost model,
// and fusion pass.
//
// Functions:
//   buildConsumerMap     — tensor id → list of consuming node ids
//   extractChainBoundary — classify tensors of a candidate chain as
//                          external-inputs / external-outputs / internal
//   createFusedNode      — build the replacement fused Node + output Tensors
// ─────────────────────────────────────────────────────────────────────────────

import { Graph, Node, Tensor } from "../ir/graph";

// Module-level counter for unique fused-node/tensor ids.
let _fusedCounter = 0;

export function resetFusedCounter(): void {
  _fusedCounter = 0;
}

// ─── Consumer Map ─────────────────────────────────────────────────────────────

/**
 * Build a map from tensor id → list of node ids that consume it.
 *
 * Every tensor in the graph receives an entry (possibly with an empty list)
 * so callers can safely use `consumers.get(tid) ?? []` without undefined checks.
 */
export function buildConsumerMap(graph: Graph): Map<string, string[]> {
  const consumers = new Map<string, string[]>();

  // Pre-fill with empty arrays so every tensor has an entry.
  for (const tid of graph.tensors.keys()) {
    consumers.set(tid, []);
  }

  for (const node of graph.nodes.values()) {
    for (const tid of node.inputs) {
      const list = consumers.get(tid);
      if (list) {
        list.push(node.id);
      } else {
        consumers.set(tid, [node.id]);
      }
    }
  }

  return consumers;
}

// ─── Chain Boundary Extraction ────────────────────────────────────────────────

export interface ChainBoundary {
  /** Tensor ids consumed by the chain but produced outside it (or graph inputs). */
  externalInputs:  string[];
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
export function extractChainBoundary(
  graph:     Graph,
  chain:     string[],
  consumers: Map<string, string[]>,
): ChainBoundary {
  const chainSet       = new Set(chain);
  const graphOutputSet = new Set(graph.outputIds);

  const externalInputs:  string[] = [];
  const externalOutputs: string[] = [];
  const internalTensors: string[] = [];

  for (const nid of chain) {
    const node = graph.getNode(nid);

    // Classify inputs: is the producer outside the chain?
    for (const tid of node.inputs) {
      const tensor = graph.getTensor(tid);
      const isExternallyProduced =
        tensor.producerNodeId === null ||     // graph input
        !chainSet.has(tensor.producerNodeId); // produced by a node outside chain
      if (isExternallyProduced && !externalInputs.includes(tid)) {
        externalInputs.push(tid);
      }
    }

    // Classify outputs: are all consumers within the chain AND not a graph output?
    for (const tid of node.outputs) {
      const cons = consumers.get(tid) ?? [];
      const hasExternalConsumer = cons.some(c => !chainSet.has(c));

      if (hasExternalConsumer || graphOutputSet.has(tid)) {
        externalOutputs.push(tid);
      } else {
        internalTensors.push(tid);
      }
    }
  }

  return { externalInputs, externalOutputs, internalTensors };
}

// ─── Fused Node Factory ───────────────────────────────────────────────────────

export interface FusedNodeResult {
  node:    Node;
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
export function createFusedNode(
  fusedOp:         string,
  externalInputs:  string[],
  externalOutputs: string[],
  graph:           Graph,
): FusedNodeResult {
  const nodeId = `n_fused_${_fusedCounter}`;

  // Mirror each external output tensor with a new fused tensor.
  const tensors: Tensor[] = externalOutputs.map((oldTid, i) => {
    const old = graph.getTensor(oldTid);
    return {
      id:             `t_fused_${_fusedCounter}_${i}`,
      name:           `${fusedOp}_out${i === 0 ? "" : `_${i}`}`,
      dtype:          old.dtype,
      shape:          [...old.shape],
      producerNodeId: nodeId,
    };
  });

  _fusedCounter++;

  const node: Node = {
    id:      nodeId,
    op:      fusedOp,
    inputs:  [...externalInputs],
    outputs: tensors.map(t => t.id),
    attrs:   { fusedFrom: [...externalOutputs] },
  };

  return { node, tensors };
}
