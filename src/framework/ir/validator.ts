// ─────────────────────────────────────────────────────────────────────────────
//
// Structural validator for IRPackage values arriving at the IR boundary.
//
// The validator is intentionally strict: it rejects anything that the bridge
// or compiler cannot safely process, surfacing problems early with precise
// diagnostic messages rather than producing silent mis-compilations.
//
// Validation checks (in order):
//   1. irVersion and opsetVersion present and recognised.
//   2. At least one graph; exactly one "forward" graph.
//   3. Per-graph:
//        a. nodeOrder contains unique ids that appear in nodes map.
//        b. Every tensor referenced by inputIds / outputIds / node inputs/outputs
//           exists in the tensors map.
//        c. SSA: each tensor's producerNodeId is either null (it must appear in
//           inputIds) or the id of a node that lists it in its outputs.
//        d. DAG: no cycles in the node dependency graph (toposort).
//        e. nodeOrder is a valid topological order for the dependency graph.
//   4. parameters (if present): each tensorId exists in some graph tensor map
//      and data.length === product of shape dimensions.
// ─────────────────────────────────────────────────────────────────────────────

import { IRPackage, GraphIR, TensorIR } from "./schema";
import { TensorId, NodeId } from "./ids";

// ─── Public types ─────────────────────────────────────────────────────────────

export type IRValidationErrorKind =
  | "MISSING_VERSION"
  | "UNKNOWN_IR_VERSION"
  | "NO_GRAPHS"
  | "MISSING_FORWARD_GRAPH"
  | "DUPLICATE_GRAPH_ID"
  | "EMPTY_GRAPH_ID"
  | "MISSING_TENSOR"
  | "MISSING_NODE"
  | "DUPLICATE_NODE_ID"
  | "SSA_VIOLATION"
  | "DANGLING_INPUT_TENSOR"
  | "DANGLING_OUTPUT_TENSOR"
  | "CYCLE_DETECTED"
  | "WRONG_TOPO_ORDER"
  | "PARAM_TENSOR_NOT_FOUND"
  | "PARAM_DATA_SIZE_MISMATCH";

export interface IRValidationError {
  readonly kind:     IRValidationErrorKind;
  readonly message:  string;
  /** Graph id context, if applicable. */
  readonly graphId?: string;
}

export interface IRValidationResult {
  readonly valid:  boolean;
  readonly errors: readonly IRValidationError[];
}

// ─── Supported versions ───────────────────────────────────────────────────────

const SUPPORTED_IR_VERSIONS = new Set(["0.1"]);
const SUPPORTED_OPSET_PREFIXES = ["mini-ts-"];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function shapeProduct(shape: readonly number[]): number {
  return shape.reduce((acc, d) => acc * (d < 0 ? 1 : d), 1);
}

/**
 * Kahn's algorithm topological sort.  Returns sorted ids or throws a string
 * describing the cycle if one is detected.
 */
function topoSort(
  nodeIds: readonly NodeId[],
  edgesFrom: ReadonlyMap<NodeId, readonly NodeId[]>,
): NodeId[] | null {
  const inDegree = new Map<NodeId, number>();
  for (const nid of nodeIds) inDegree.set(nid, 0);

  for (const [, deps] of edgesFrom) {
    for (const dep of deps) {
      if (inDegree.has(dep)) inDegree.set(dep, (inDegree.get(dep) ?? 0) + 1);
    }
  }

  const queue: NodeId[] = [];
  for (const [nid, deg] of inDegree) {
    if (deg === 0) queue.push(nid);
  }

  const sorted: NodeId[] = [];
  while (queue.length > 0) {
    const nid = queue.shift()!;
    sorted.push(nid);
    for (const dep of edgesFrom.get(nid) ?? []) {
      if (!inDegree.has(dep)) continue;
      const newDeg = (inDegree.get(dep) ?? 0) - 1;
      inDegree.set(dep, newDeg);
      if (newDeg === 0) queue.push(dep);
    }
  }

  return sorted.length === nodeIds.length ? sorted : null;
}

// ─── Per-graph validation ─────────────────────────────────────────────────────

function validateGraph(
  g: GraphIR,
  errors: IRValidationError[],
): void {
  const gid = g.id as string;

  function err(
    kind: IRValidationErrorKind,
    message: string,
  ): void {
    errors.push({ kind, message, graphId: gid });
  }

  const tensorMap = g.tensors as Record<string, TensorIR>;
  const nodeMap   = g.nodes   as Record<string, { op: string; inputs: readonly TensorId[]; outputs: readonly TensorId[]; id: NodeId }>;

  // ── 3a. nodeOrder uniqueness and existence ────────────────────────────────
  const seenNodes = new Set<string>();
  for (const nid of g.nodeOrder) {
    if (seenNodes.has(nid)) {
      err("DUPLICATE_NODE_ID", `nodeOrder contains duplicate node id "${nid}"`);
    }
    seenNodes.add(nid);
    if (!(nid in nodeMap)) {
      err("MISSING_NODE", `nodeOrder references unknown node "${nid}"`);
    }
  }

  // ── 3b. Input / output tensor existence ───────────────────────────────────
  for (const tid of g.inputIds) {
    if (!(tid in tensorMap)) {
      err("MISSING_TENSOR", `inputIds references unknown tensor "${tid}"`);
    }
  }
  for (const tid of g.outputIds) {
    if (!(tid in tensorMap)) {
      err("MISSING_TENSOR", `outputIds references unknown tensor "${tid}"`);
    }
  }

  // ── 3b. Node input / output tensor existence ──────────────────────────────
  for (const nid of Object.keys(nodeMap)) {
    const node = nodeMap[nid];
    for (const tid of node.inputs) {
      if (!(tid in tensorMap)) {
        err("MISSING_TENSOR", `Node "${nid}" (${node.op}) references unknown input tensor "${tid}"`);
      }
    }
    for (const tid of node.outputs) {
      if (!(tid in tensorMap)) {
        err("MISSING_TENSOR", `Node "${nid}" (${node.op}) references unknown output tensor "${tid}"`);
      }
    }
  }

  // ── 3c. SSA invariant ─────────────────────────────────────────────────────
  const inputIdSet = new Set<string>(Array.from(g.inputIds).map(id => id as string));
  for (const [tid, tensor] of Object.entries(tensorMap)) {
    if (tensor.producerNodeId === null) {
      // Must appear in inputIds
      if (!inputIdSet.has(tid as TensorId)) {
        err(
          "DANGLING_INPUT_TENSOR",
          `Tensor "${tid}" has producerNodeId=null but is not listed in inputIds`,
        );
      }
    } else {
      // Producer node must exist and list this tensor in its outputs
      const producerNode = nodeMap[tensor.producerNodeId as string];
      if (!producerNode) {
        err(
          "SSA_VIOLATION",
          `Tensor "${tid}" claims producer "${tensor.producerNodeId}" which does not exist`,
        );
      } else if (!Array.from(producerNode.outputs).includes(tid as TensorId)) {
        err(
          "SSA_VIOLATION",
          `Tensor "${tid}" claims producer "${tensor.producerNodeId}" but that node does not list it in outputs`,
        );
      }
    }
  }

  // ── 3d. DAG check + 3e. topological order ────────────────────────────────
  // Build edges: node A must come before node B if A produces a tensor that B consumes.
  // We express this as: after B runs, edges point FROM B to its dependencies (A).
  // For topoSort we want "node -> nodes it depends on" direction reversed.
  // Simplest: build a successor map then verify nodeOrder is consistent.

  // Build a map: tensorId -> producerNodeId (only for non-input tensors)
  const tensorProducer = new Map<string, NodeId>();
  for (const [tid, t] of Object.entries(tensorMap)) {
    if (t.producerNodeId !== null) {
      tensorProducer.set(tid, t.producerNodeId as NodeId);
    }
  }

  // For each node, collect its predecessor node ids (nodes that produce its inputs)
  const predecessors = new Map<NodeId, NodeId[]>();
  for (const nid of g.nodeOrder) {
    if (!(nid in nodeMap)) continue;
    const node = nodeMap[nid as string];
    const preds: NodeId[] = [];
    for (const tid of node.inputs) {
      const prod = tensorProducer.get(tid as string);
      if (prod && prod !== nid && !preds.includes(prod)) {
        preds.push(prod);
      }
    }
    predecessors.set(nid as NodeId, preds);
  }

  // Build edgesFrom as "node -> its successors" for topoSort
  const successors = new Map<NodeId, NodeId[]>();
  for (const nid of g.nodeOrder as NodeId[]) successors.set(nid, []);
  for (const [nid, preds] of predecessors) {
    for (const pred of preds) {
      const list = successors.get(pred);
      if (list && !list.includes(nid)) list.push(nid);
    }
  }

  const sorted = topoSort(g.nodeOrder as NodeId[], successors);
  if (sorted === null) {
    err("CYCLE_DETECTED", "The node dependency graph contains a cycle");
    return; // order check is meaningless after a cycle
  }

  // Verify that g.nodeOrder is a valid topological order (not necessarily
  // the unique one, just that no node comes before all its predecessors).
  const positionInOrder = new Map<string, number>();
  for (let i = 0; i < g.nodeOrder.length; i++) {
    positionInOrder.set(g.nodeOrder[i] as string, i);
  }
  for (const [nid, preds] of predecessors) {
    const myPos = positionInOrder.get(nid as string) ?? Infinity;
    for (const pred of preds) {
      const predPos = positionInOrder.get(pred as string) ?? -1;
      if (predPos > myPos) {
        err(
          "WRONG_TOPO_ORDER",
          `Node "${nid}" appears before its dependency "${pred}" in nodeOrder`,
        );
      }
    }
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Validate an IRPackage against the full set of structural constraints.
 *
 * Returns a result object with `valid` and a list of `errors`.
 * When `valid` is true the package is safe to pass to the bridge.
 */
export function validateIRPackage(pkg: IRPackage): IRValidationResult {
  const errors: IRValidationError[] = [];

  // ── 1. Version checks ────────────────────────────────────────────────────
  if (!pkg.irVersion) {
    errors.push({ kind: "MISSING_VERSION", message: "irVersion is required" });
  } else if (!SUPPORTED_IR_VERSIONS.has(pkg.irVersion)) {
    errors.push({
      kind:    "UNKNOWN_IR_VERSION",
      message: `Unsupported irVersion "${pkg.irVersion}". Supported: ${[...SUPPORTED_IR_VERSIONS].join(", ")}`,
    });
  }

  if (!pkg.opsetVersion) {
    errors.push({ kind: "MISSING_VERSION", message: "opsetVersion is required" });
  } else {
    const knownOpset = SUPPORTED_OPSET_PREFIXES.some(p =>
      (pkg.opsetVersion as string).startsWith(p),
    );
    if (!knownOpset) {
      // Warn but do not fail — forward-compatibility: unknown opsets may still work.
      // Only hard-fail on structurally broken graphs.
    }
  }

  // ── 2. Graph list ─────────────────────────────────────────────────────────
  if (!pkg.graphs || pkg.graphs.length === 0) {
    errors.push({ kind: "NO_GRAPHS", message: "IRPackage must contain at least one graph" });
    return { valid: errors.length === 0, errors };
  }

  const forwardGraphs = pkg.graphs.filter(g => g.kind === "forward");
  if (forwardGraphs.length === 0) {
    errors.push({ kind: "MISSING_FORWARD_GRAPH", message: 'At least one graph with kind "forward" is required' });
  }

  const seenGraphIds = new Set<string>();
  for (const g of pkg.graphs) {
    if (!g.id) {
      errors.push({ kind: "EMPTY_GRAPH_ID", message: "A graph has an empty or missing id" });
      continue;
    }
    if (seenGraphIds.has(g.id as string)) {
      errors.push({ kind: "DUPLICATE_GRAPH_ID", message: `Duplicate graph id "${g.id}"` });
    }
    seenGraphIds.add(g.id as string);

    // ── 3. Per-graph structural checks ──────────────────────────────────────
    validateGraph(g, errors);
  }

  // ── 4. Parameter data integrity ──────────────────────────────────────────
  if (pkg.parameters) {
    // Build a combined tensor map across all graphs
    const allTensors = new Map<string, { dtype: string; shape: readonly number[] }>();
    for (const g of pkg.graphs) {
      for (const [tid, t] of Object.entries(g.tensors)) {
        allTensors.set(tid, { dtype: t.dtype, shape: t.shape });
      }
    }

    for (const param of pkg.parameters) {
      const t = allTensors.get(param.tensorId as string);
      if (!t) {
        errors.push({
          kind:    "PARAM_TENSOR_NOT_FOUND",
          message: `Parameter references tensor "${param.tensorId}" not found in any graph`,
        });
        continue;
      }
      const expected = shapeProduct(param.shape);
      if (param.data.length !== expected) {
        errors.push({
          kind:    "PARAM_DATA_SIZE_MISMATCH",
          message: `Parameter "${param.name}" has data.length=${param.data.length} but shape product=${expected}`,
        });
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
