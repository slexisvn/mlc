// ─────────────────────────────────────────────────────────────────────────────
// Pure graph analysis for reverse-mode autodiff.
//
// Algorithm
// ─────────
// A forward node requires gradient computation if and only if it lies on a
// path that both:
//   (a) is reachable *forward* from at least one paramId, AND
//   (b) is reachable *backward* from at least one rootId (loss tensor).
//
// Nodes satisfying only (a) are "dead-end" branches — reachable from a
// parameter but not contributing to any root.  Their gradients are
// mathematically zero and can be skipped entirely.
//
// Nodes satisfying only (b) are "root-only" branches — part of the loss
// computation but not depending on any differentiable parameter.  They
// still don't require grad computation.
// ─────────────────────────────────────────────────────────────────────────────

import { GraphIR }          from "../ir/schema";
import { TensorId, NodeId } from "../ir/ids";

// ─── Result type ──────────────────────────────────────────────────────────────

export interface AutodiffAnalysis {
  /**
   * Forward nodes that lie on at least one param→root path.
   * Only these nodes will have gradient builders invoked.
   */
  readonly neededNodes: ReadonlySet<NodeId>;

  /** tensor-id → id of the node that produced it (computed once, shared). */
  readonly tensorToNode: ReadonlyMap<TensorId, NodeId>;

  /** tensor-id → ids of every node that consumes it. */
  readonly tensorToConsumers: ReadonlyMap<TensorId, readonly NodeId[]>;
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute the set of forward nodes that must be differentiated.
 *
 * Returns an `AutodiffAnalysis` that also exposes the adjacency maps so the
 * caller (BackwardGraphBuilder) does not need to rebuild them.
 */
export function analyzeAutodiff(
  fwd:      GraphIR,
  paramIds: readonly TensorId[],
  rootIds:  readonly TensorId[],
): AutodiffAnalysis {

  // ── Build tensor↔node adjacency maps ─────────────────────────────────────

  const tensorToNode      = new Map<TensorId, NodeId>();
  const tensorToConsumers = new Map<TensorId, NodeId[]>();

  for (const nid of fwd.nodeOrder) {
    const node = fwd.nodes[nid];
    for (const tid of node.outputs) tensorToNode.set(tid, nid);
    for (const tid of node.inputs) {
      if (!tensorToConsumers.has(tid)) tensorToConsumers.set(tid, []);
      tensorToConsumers.get(tid)!.push(nid);
    }
  }

  // ── (a) Forward reachability: params → any descendant node ───────────────

  const fwdReachable = new Set<NodeId>();
  const seenFwd      = new Set<TensorId>(paramIds);

  function markForward(tid: TensorId): void {
    for (const nid of tensorToConsumers.get(tid) ?? []) {
      if (fwdReachable.has(nid)) continue;
      fwdReachable.add(nid);
      for (const outId of fwd.nodes[nid].outputs) {
        if (!seenFwd.has(outId)) { seenFwd.add(outId); markForward(outId); }
      }
    }
  }
  for (const pid of paramIds) markForward(pid);

  // ── (b) Backward reachability: roots → any ancestor node ─────────────────

  const bwdReachable = new Set<NodeId>();
  const seenBwd      = new Set<TensorId>(rootIds);

  function markBackward(tid: TensorId): void {
    const nid = tensorToNode.get(tid);
    if (!nid || bwdReachable.has(nid)) return;
    bwdReachable.add(nid);
    for (const inId of fwd.nodes[nid].inputs) {
      if (!seenBwd.has(inId)) { seenBwd.add(inId); markBackward(inId); }
    }
  }
  for (const rid of rootIds) markBackward(rid);

  // ── Intersection: nodes on both paths ─────────────────────────────────────

  const neededNodes = new Set<NodeId>();
  for (const nid of fwdReachable) {
    if (bwdReachable.has(nid)) neededNodes.add(nid);
  }

  return { neededNodes, tensorToNode, tensorToConsumers };
}
