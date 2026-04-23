// ─────────────────────────────────────────────────────────────────────────────
//
// Initialiser functions and ParameterSpec type.
// ─────────────────────────────────────────────────────────────────────────────

import { ShapeExpr, shapeNumel } from "../core/shape";
import { IRDType }               from "../ir/schema";
import { SymbolicTensor }        from "../tensor/tensor";

// ─── Re-export ParamSpec under the public name ParameterSpec ─────────────────
export type { ParamSpec as ParameterSpec } from "../core/context";

// ─── Initialiser ─────────────────────────────────────────────────────────────

/** A weight initialiser: receives shape, returns flat row-major values. */
export type Initialiser = (shape: ShapeExpr) => number[];

/** Glorot / Xavier uniform initialiser. */
export function initXavier(shape: ShapeExpr): number[] {
  const n = shapeNumel(shape);
  if (n <= 0) return [];
  const fanIn  = shape.length >= 2 ? shape[shape.length - 2] : shape[0];
  const fanOut = shape[shape.length - 1] ?? fanIn;
  const limit  = Math.sqrt(6 / (fanIn + fanOut));
  return Array.from({ length: n }, () => (Math.random() * 2 - 1) * limit);
}

/** Zero initialiser. */
export function initZeros(shape: ShapeExpr): number[] {
  const n = shapeNumel(shape);
  return Array(n < 0 ? 0 : n).fill(0);
}

/** Ones initialiser. */
export function initOnes(shape: ShapeExpr): number[] {
  const n = shapeNumel(shape);
  return Array(n < 0 ? 0 : n).fill(1);
}

/** Constant initialiser factory. */
export function initConstant(value: number): Initialiser {
  return (shape: ShapeExpr) => {
    const n = shapeNumel(shape);
    return Array(n < 0 ? 0 : n).fill(value);
  };
}
