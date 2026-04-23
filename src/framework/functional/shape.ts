import { getActiveBuilder } from "../core/context";
import { SymbolicTensor }   from "../tensor/tensor";

/**
 * Reshape a tensor.
 * Exactly one element of `shape` may be -1 (inferred from element count).
 */
export function reshape(x: SymbolicTensor, shape: number[]): SymbolicTensor {
  return getActiveBuilder().applyOp(
    "reshape", [x], { shape } as unknown as Record<string, unknown>,
  )[0];
}

/**
 * Transpose (permute) tensor dimensions.
 * `perm` defaults to reversing all dimensions.
 */
export function transpose(x: SymbolicTensor, perm?: number[]): SymbolicTensor {
  const attrs = perm
    ? ({ perm } as unknown as Record<string, unknown>)
    : {};
  return getActiveBuilder().applyOp("transpose", [x], attrs)[0];
}
