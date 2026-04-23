// ─────────────────────────────────────────────────────────────────────────────
// Graph-invariant checker.
//
// Invariants verified:
//   1. Dangling edges  — every tensor id in node.inputs/outputs exists.
//   2. SSA property    — each non-input tensor has exactly one producer node,
//                        and that node's outputs list includes the tensor.
//   3. Graph outputs   — all declared output tensor ids exist.
//   4. DAG property    — no cycles (Kahn's topological sort).
//
// Called by the PassManager after each pass when validateAfterEachPass=true.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph } from "./graph";

export type ValidationErrorKind =
  | "DanglingEdge"
  | "SSAViolation"
  | "MissingOutput"
  | "Cycle"
  | "OrphanNode"
  | "LayoutContractViolation";

export interface ValidationError {
  kind:    ValidationErrorKind;
  message: string;
}

export interface ValidationResult {
  valid:  boolean;
  errors: ValidationError[];
}

export function validateGraph(graph: Graph): ValidationResult {
  const errors: ValidationError[] = [];

  // ── Check 1: Dangling edges ────────────────────────────────────────────────
  for (const node of graph.nodes.values()) {
    for (const tid of node.inputs) {
      if (!graph.tensors.has(tid)) {
        errors.push({
          kind:    "DanglingEdge",
          message: `Node "${node.id}" (${node.op}) references non-existent input tensor "${tid}"`,
        });
      }
    }
    for (const tid of node.outputs) {
      if (!graph.tensors.has(tid)) {
        errors.push({
          kind:    "DanglingEdge",
          message: `Node "${node.id}" (${node.op}) references non-existent output tensor "${tid}"`,
        });
      }
    }
  }

  // ── Check 2: SSA property ──────────────────────────────────────────────────
  // Count how many times each tensor id appears in node output lists.
  const nodeProducerCount = new Map<string, number>();
  for (const node of graph.nodes.values()) {
    for (const tid of node.outputs) {
      nodeProducerCount.set(tid, (nodeProducerCount.get(tid) ?? 0) + 1);
    }
  }

  for (const tensor of graph.tensors.values()) {
    if (tensor.producerNodeId === null) {
      // Graph input: must NOT appear in any node's output list.
      if (nodeProducerCount.has(tensor.id)) {
        errors.push({
          kind:    "SSAViolation",
          message: `Graph-input tensor "${tensor.id}" (${tensor.name}) is unexpectedly produced by a node`,
        });
      }
    } else {
      // Non-input tensor: must be produced by exactly one node.
      const count = nodeProducerCount.get(tensor.id) ?? 0;
      if (count !== 1) {
        errors.push({
          kind:    "SSAViolation",
          message: `Tensor "${tensor.id}" (${tensor.name}) has ${count} producers; expected exactly 1`,
        });
      }

      // The tensor's declared producerNodeId must match the actual producing node.
      const producerNode = graph.nodes.get(tensor.producerNodeId);
      if (!producerNode) {
        errors.push({
          kind:    "DanglingEdge",
          message: `Tensor "${tensor.id}" declares producerNodeId "${tensor.producerNodeId}" but that node does not exist`,
        });
      } else if (!producerNode.outputs.includes(tensor.id)) {
        errors.push({
          kind:    "SSAViolation",
          message: `Tensor "${tensor.id}" declares producer "${tensor.producerNodeId}" but that node does not list it as an output`,
        });
      }
    }
  }

  // ── Check 3: Graph outputs exist ──────────────────────────────────────────
  for (const oid of graph.outputIds) {
    if (!graph.tensors.has(oid)) {
      errors.push({
        kind:    "MissingOutput",
        message: `Graph output tensor "${oid}" does not exist`,
      });
    }
  }

  // ── Check 4: DAG property (Kahn's algorithm) ──────────────────────────────
  const inDegree = new Map<string, number>();
  const adjOut   = new Map<string, string[]>();

  for (const nid of graph.nodeOrder) {
    inDegree.set(nid, 0);
    adjOut.set(nid, []);
  }

  // Build tensor-id → producing-node-id index.
  const tensorProducer = new Map<string, string>();
  for (const node of graph.nodes.values()) {
    for (const tid of node.outputs) tensorProducer.set(tid, node.id);
  }

  // Collect unique directed edges to avoid multi-edge in-degree miscounts.
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

  const queue: string[] = [];
  for (const [nid, deg] of inDegree) {
    if (deg === 0) queue.push(nid);
  }
  queue.sort(); // deterministic

  let visited = 0;
  while (queue.length > 0) {
    queue.sort();
    const cur = queue.shift()!;
    visited++;
    for (const next of (adjOut.get(cur) ?? [])) {
      const newDeg = (inDegree.get(next) ?? 1) - 1;
      inDegree.set(next, newDeg);
      if (newDeg === 0) queue.push(next);
    }
  }

  if (visited < graph.nodeOrder.length) {
    errors.push({
      kind:    "Cycle",
      message: `Graph contains a cycle (processed ${visited}/${graph.nodeOrder.length} nodes)`,
    });
  }

  return { valid: errors.length === 0, errors };
}
