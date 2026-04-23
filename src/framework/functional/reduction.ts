// ─────────────────────────────────────────────────────────────────────────────
// frontend/functional/reduction.ts
//
// Reduction ops.
// ─────────────────────────────────────────────────────────────────────────────

import { getActiveBuilder } from "../core/context";
import { SymbolicTensor }   from "../tensor/tensor";

export interface ReduceOptions {
  /** Axes to reduce. Defaults to all axes. */
  axes?:     number[];
  /** Keep reduced dimensions as size-1. Defaults to false. */
  keepDims?: boolean;
}

/** Sum elements along the specified axes. */
export function sum(x: SymbolicTensor, options: ReduceOptions = {}): SymbolicTensor {
  const axes     = options.axes     ?? Array.from({ length: x.rank }, (_, i) => i);
  const keepDims = options.keepDims ?? false;
  return getActiveBuilder().applyOp("sum", [x], { axes, keepDims })[0];
}

/** Mean of elements along the specified axes. */
export function mean(x: SymbolicTensor, options: ReduceOptions = {}): SymbolicTensor {
  const axes     = options.axes     ?? Array.from({ length: x.rank }, (_, i) => i);
  const keepDims = options.keepDims ?? false;
  return getActiveBuilder().applyOp("mean", [x], { axes, keepDims })[0];
}
