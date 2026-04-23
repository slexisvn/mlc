// ─────────────────────────────────────────────────────────────────────────────
// frontend/functional/linear.ts
//
// Linear algebra functional ops.
// ─────────────────────────────────────────────────────────────────────────────

import { getActiveBuilder } from "../core/context";
import { SymbolicTensor }   from "../tensor/tensor";

/**
 * Matrix multiplication: a @ b.
 * Supports 2-D and batched matmul.
 */
export function matmul(a: SymbolicTensor, b: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("matmul", [a, b])[0];
}

/**
 * Linear transform: x @ weight + bias (bias optional).
 * Equivalent to PyTorch's `F.linear`.
 */
export function linear(
  x:      SymbolicTensor,
  weight: SymbolicTensor,
  bias?:  SymbolicTensor,
): SymbolicTensor {
  const gb = getActiveBuilder();
  const [mm] = gb.applyOp("matmul", [x, weight]);
  if (bias) {
    const [out] = gb.applyOp("add", [mm, bias]);
    return out;
  }
  return mm;
}
