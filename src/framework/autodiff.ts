// ─────────────────────────────────────────────────────────────────────────────
// frontend/autodiff.ts
//
// Reverse-mode automatic differentiation over a GraphIR.
// Move-only from framework/autodiff.ts — logic is unchanged.
// ─────────────────────────────────────────────────────────────────────────────

import { GraphIR, IRAttrs, IRDType }              from "./ir/schema";
import { TensorId, NodeId }                       from "./ir/ids";
import { AutodiffError }                          from "./core/errors";
import { OpSchemaRegistry, defaultOpRegistry, GradBuilderFn } from "./core/opRegistry";
import { SymbolicTensor }                         from "./tensor/tensor";
import { GraphBuilder }                           from "./core/graphBuilder";

// ─── Result type ──────────────────────────────────────────────────────────────

export interface BackwardResult {
  readonly backwardGraph: GraphIR;
  /** forward-param-id → backward-gradient-id */
  readonly gradMap: ReadonlyMap<TensorId, TensorId>;
}

// ─── Core algorithm ───────────────────────────────────────────────────────────

export function buildBackwardGraph(
  fwd:      GraphIR,
  rootIds:  readonly TensorId[],
  paramIds: readonly TensorId[],
  registry: OpSchemaRegistry = defaultOpRegistry,
): BackwardResult {
  // tensor → producing node
  const tensorToNode = new Map<TensorId, NodeId>();
  for (const nid of fwd.nodeOrder) {
    for (const tid of fwd.nodes[nid].outputs) tensorToNode.set(tid, nid);
  }

  // tensor → consuming nodes
  const tensorToConsumers = new Map<TensorId, NodeId[]>();
  for (const nid of fwd.nodeOrder) {
    for (const tid of fwd.nodes[nid].inputs) {
      if (!tensorToConsumers.has(tid)) tensorToConsumers.set(tid, []);
      tensorToConsumers.get(tid)!.push(nid);
    }
  }

  // Determine nodes reachable forward from params → roots
  const neededTensors = new Set<TensorId>(paramIds);
  const neededNodes   = new Set<NodeId>();
  function markForward(tid: TensorId): void {
    if (!neededTensors.has(tid)) neededTensors.add(tid);
    for (const nid of tensorToConsumers.get(tid) ?? []) {
      if (neededNodes.has(nid)) continue;
      neededNodes.add(nid);
      for (const outId of fwd.nodes[nid].outputs) markForward(outId);
    }
  }
  for (const pid of paramIds) markForward(pid);

  // Intersect with nodes backward-reachable from roots.
  // Only nodes on the param→root path need gradients; dead-end branches
  // (forward-reachable from params but not reaching any root) are excluded.
  const rootReachable = new Set<TensorId>(rootIds);
  const rootNodes     = new Set<NodeId>();
  function markBackwardReachable(tid: TensorId): void {
    if (rootReachable.has(tid)) return;
    rootReachable.add(tid);
    const nid = tensorToNode.get(tid);
    if (!nid) return;
    rootNodes.add(nid);
    for (const inId of fwd.nodes[nid].inputs) markBackwardReachable(inId);
  }
  for (const rid of rootIds) {
    rootReachable.add(rid);
    const nid = tensorToNode.get(rid);
    if (nid) {
      rootNodes.add(nid);
      for (const inId of fwd.nodes[nid].inputs) markBackwardReachable(inId);
    }
  }

  // Only differentiate nodes on both paths (intersection).
  for (const nid of [...neededNodes]) {
    if (!rootNodes.has(nid)) neededNodes.delete(nid);
  }

  // Build backward graph
  const bwdGb = new GraphBuilder({ id: "backward", opRegistry: registry });

  const fwdIdToBwdHandle    = new Map<TensorId, SymbolicTensor>();
  const fwdComputedHandles  = new Map<TensorId, SymbolicTensor>();

  // Mirror forward inputs
  for (const tid of fwd.inputIds) {
    const t = fwd.tensors[tid];
    fwdIdToBwdHandle.set(tid, bwdGb.input(t.name, t.dtype as IRDType, [...t.shape]));
  }

  // Re-run forward ops in backward builder
  for (const nid of fwd.nodeOrder) {
    const node         = fwd.nodes[nid];
    const inputHandles = node.inputs.map(tid => {
      const h = fwdIdToBwdHandle.get(tid) ?? fwdComputedHandles.get(tid);
      if (!h) throw new AutodiffError(`No backward handle for forward tensor "${tid}"`);
      return h;
    });
    const outHandles = bwdGb.applyOp(node.op, inputHandles, node.attrs as IRAttrs);
    for (let i = 0; i < node.outputs.length; i++) {
      fwdComputedHandles.set(node.outputs[i], outHandles[i]);
    }
  }

  // Seed gradients
  const gradHandles = new Map<TensorId, SymbolicTensor>();
  for (const rootId of rootIds) {
    const t = fwd.tensors[rootId];
    gradHandles.set(rootId, bwdGb.input(`grad_${t.name}`, t.dtype as IRDType, [...t.shape]));
  }

  function getGrad(tid: TensorId): SymbolicTensor | undefined { return gradHandles.get(tid); }
  function addGrad(tid: TensorId, g: SymbolicTensor): void {
    const existing = gradHandles.get(tid);
    if (!existing) { gradHandles.set(tid, g); }
    else {
      const [sum] = bwdGb.applyOp("add", [existing, g]);
      gradHandles.set(tid, sum);
    }
  }
  function getBwdHandle(fwdTid: TensorId): SymbolicTensor {
    const h = fwdIdToBwdHandle.get(fwdTid) ?? fwdComputedHandles.get(fwdTid);
    if (!h) throw new AutodiffError(`No backward-side handle for forward tensor "${fwdTid}"`);
    return h;
  }
  function applyOpForGrad(op: string, inputIds: readonly string[], attrs: IRAttrs = {}): string[] {
    const inputs = inputIds.map(id => bwdGb.getTensorHandle(id as TensorId));
    return bwdGb.applyOp(op, inputs, attrs).map(t => t.id as string);
  }

  // Backward pass in reverse topo order
  for (const nid of [...fwd.nodeOrder].reverse()) {
    const node = fwd.nodes[nid];
    if (!neededNodes.has(nid)) continue;

    const outputGrads: SymbolicTensor[] = [];
    for (const outId of node.outputs) {
      const g = getGrad(outId);
      if (!g) throw new AutodiffError(`No gradient for output "${outId}" of node "${nid}" (${node.op})`);
      outputGrads.push(g);
    }

    const schema = registry.has(node.op) ? registry.get(node.op) : undefined;
    if (!schema?.gradBuilder) {
      throw new AutodiffError(`Op "${node.op}" has no gradient builder — node "${nid}"`);
    }

    const outputIds     = node.outputs.map(id => getBwdHandle(id).id as string);
    const gradIds       = outputGrads.map(g => g.id as string);
    const fwdCtx        = {
      inputShapes:  node.inputs.map(tid  => [...fwd.tensors[tid].shape]),
      inputDtypes:  node.inputs.map(tid  => fwd.tensors[tid].dtype as IRDType),
      attrs:        node.attrs as IRAttrs,
    };
    const fwdInputBwdIds = node.inputs.map(tid => getBwdHandle(tid).id as string);
    const inputGradIds  = schema.gradBuilder(fwdCtx, outputIds, gradIds, applyOpForGrad, fwdInputBwdIds);

    for (let i = 0; i < node.inputs.length; i++) {
      const gradId = inputGradIds[i];
      if (gradId) addGrad(node.inputs[i], bwdGb.getTensorHandle(gradId as TensorId));
    }
  }

  // Collect param gradients
  const gradMap               = new Map<TensorId, TensorId>();
  const gradOutputHandles: SymbolicTensor[] = [];
  for (const pid of paramIds) {
    const g = getGrad(pid);
    if (!g) throw new AutodiffError(`No gradient for parameter "${pid}"`);
    gradMap.set(pid, g.id);
    gradOutputHandles.push(g);
  }

  bwdGb.markOutputs(...gradOutputHandles);
  return { backwardGraph: bwdGb.build("backward"), gradMap };
}

// ─── Default grad builders ────────────────────────────────────────────────────

export const DEFAULT_GRAD_BUILDERS: Record<string, GradBuilderFn> = {
  add(_,   _outs, gradIds)         { return [gradIds[0], gradIds[0]]; },
  sub(_,   _outs, gradIds, apply)  { const [n] = apply("neg", [gradIds[0]]); return [gradIds[0], n]; },
  mul(_,   _outs, gradIds)         { return [gradIds[0], gradIds[0]]; },   // v1 placeholder
  relu(_,  _outs, gradIds)         { return [gradIds[0]]; },               // v1 placeholder
  sum(_,   _outs, gradIds)         { return [gradIds[0]]; },               // v1 placeholder
  matmul(_ctx, _outs, gradIds, apply, inputIds) {
    // dX = dY @ W^T
    const [wT] = apply("transpose", [inputIds[1]]);
    const [dX] = apply("matmul", [gradIds[0], wT]);
    // dW = X^T @ dY
    const [xT] = apply("transpose", [inputIds[0]]);
    const [dW] = apply("matmul", [xT, gradIds[0]]);
    return [dX, dW];
  },
};
