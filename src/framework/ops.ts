// ─────────────────────────────────────────────────────────────────────────────
// framework/ops.ts
//
// Functional op API for the framework frontend.
//
// Each function here is a thin wrapper around GraphBuilder.applyOp() that
// provides a typed, ergonomic interface:
//   • Named positional parameters instead of a generic inputs array.
//   • Op-specific attribute types (e.g. ReshapeAttrs, TransposeAttrs).
//   • Single-output ops return a SymbolicTensor directly (not an array).
//
// All functions require a GraphBuilder to be passed explicitly.  This keeps
// the API stateless and avoids global state; modules and user code always
// have an explicit dependency on a specific graph.
//
// Usage:
//   const gb = new GraphBuilder();
//   const x  = gb.input("x", "float32", [32, 784]);
//   const w  = gb.param("w", "float32", [784, 256]);
//   const y  = matmul(gb, x, w);
//   const z  = relu(gb, y);
// ─────────────────────────────────────────────────────────────────────────────

import { GraphBuilder }   from "./graphBuilder";
import { SymbolicTensor } from "./tensor";

// ─── Binary elementwise ops ───────────────────────────────────────────────────

/** Element-wise addition: x + y (supports broadcasting). */
export function add(gb: GraphBuilder, x: SymbolicTensor, y: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("add", [x, y])[0];
}

/** Element-wise subtraction: x - y (supports broadcasting). */
export function sub(gb: GraphBuilder, x: SymbolicTensor, y: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("sub", [x, y])[0];
}

/** Element-wise multiplication: x * y (supports broadcasting). */
export function mul(gb: GraphBuilder, x: SymbolicTensor, y: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("mul", [x, y])[0];
}

/** Element-wise division: x / y (supports broadcasting). */
export function div(gb: GraphBuilder, x: SymbolicTensor, y: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("div", [x, y])[0];
}

// ─── Unary elementwise ops ────────────────────────────────────────────────────

/** Rectified linear unit: max(0, x). */
export function relu(gb: GraphBuilder, x: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("relu", [x])[0];
}

/** Sigmoid activation: 1 / (1 + exp(-x)). */
export function sigmoid(gb: GraphBuilder, x: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("sigmoid", [x])[0];
}

/** Hyperbolic tangent. */
export function tanh(gb: GraphBuilder, x: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("tanh", [x])[0];
}

/** Gaussian error linear unit. */
export function gelu(gb: GraphBuilder, x: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("gelu", [x])[0];
}

/** Element-wise exponential: e^x. */
export function exp(gb: GraphBuilder, x: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("exp", [x])[0];
}

/** Element-wise square root. */
export function sqrt(gb: GraphBuilder, x: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("sqrt", [x])[0];
}

/** Element-wise negation: -x. */
export function neg(gb: GraphBuilder, x: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("neg", [x])[0];
}

/** Element-wise absolute value. */
export function abs(gb: GraphBuilder, x: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("abs", [x])[0];
}

// ─── Linear algebra ops ───────────────────────────────────────────────────────

/**
 * Matrix multiplication.
 *
 * Supports 2-D ([M,K] × [K,N] → [M,N]) and batched matmul
 * ([...,M,K] × [...,K,N] → [...,M,N]).
 */
export function matmul(gb: GraphBuilder, a: SymbolicTensor, b: SymbolicTensor): SymbolicTensor {
  return gb.applyOp("matmul", [a, b])[0];
}

// ─── Shape manipulation ───────────────────────────────────────────────────────

export interface ReshapeAttrs {
  /** Target shape.  Exactly one element may be -1 (inferred from element count). */
  shape: number[];
}

/**
 * Reshape a tensor to a new shape.
 *
 * @example
 * const flat = reshape(gb, x, { shape: [-1, 784] });
 */
export function reshape(
  gb:    GraphBuilder,
  x:     SymbolicTensor,
  attrs: ReshapeAttrs,
): SymbolicTensor {
  return gb.applyOp("reshape", [x], attrs as unknown as Record<string, unknown>)[0];
}

export interface TransposeAttrs {
  /**
   * Permutation array.  Length must equal the tensor rank.
   * Defaults to reversing all dimensions if omitted.
   */
  perm?: number[];
}

/**
 * Transpose (permute) the dimensions of a tensor.
 *
 * @example
 * const xt = transpose(gb, x, { perm: [0, 2, 1] });
 */
export function transpose(
  gb:    GraphBuilder,
  x:     SymbolicTensor,
  attrs: TransposeAttrs = {},
): SymbolicTensor {
  return gb.applyOp("transpose", [x], attrs as unknown as Record<string, unknown>)[0];
}

// ─── Reduction ops ────────────────────────────────────────────────────────────

export interface ReduceAttrs {
  /** Axes to reduce.  Defaults to all axes (global reduction). */
  axes?:     number[];
  /** When true, keep reduced axes as size-1.  Defaults to false. */
  keepDims?: boolean;
}

/** Sum along the specified axes. */
export function sum(gb: GraphBuilder, x: SymbolicTensor, attrs: ReduceAttrs = {}): SymbolicTensor {
  const axes     = attrs.axes     ?? Array.from({ length: x.rank }, (_, i) => i);
  const keepDims = attrs.keepDims ?? false;
  return gb.applyOp("sum", [x], { axes, keepDims })[0];
}

/** Mean along the specified axes. */
export function mean(gb: GraphBuilder, x: SymbolicTensor, attrs: ReduceAttrs = {}): SymbolicTensor {
  const axes     = attrs.axes     ?? Array.from({ length: x.rank }, (_, i) => i);
  const keepDims = attrs.keepDims ?? false;
  return gb.applyOp("mean", [x], { axes, keepDims })[0];
}
