// ─────────────────────────────────────────────────────────────────────────────
// framework/shape.ts
//
// Symbolic and concrete shape utilities used by the framework frontend.
//
// A ShapeExpr is a mixed array of concrete integers and symbolic dimension
// names.  During graph construction concrete shapes are preferred; symbolic
// names are used when a dimension depends on runtime data (e.g. batch size).
//
// Shape inference helpers follow NumPy broadcasting conventions:
//   broadcast(a, b) — computes the broadcast-compatible output shape or
//                      throws ShapeError for incompatible shapes.
//   matmulShape(a, b) — shape inference for 2-D and batched matmul.
//   transposeShape  — shape after applying a permutation.
//   reshapeShape    — validates -1 inference for reshape.
//
// All functions are pure: they never mutate their arguments.
// ─────────────────────────────────────────────────────────────────────────────

import { ShapeError } from "./errors";

// ─── Types ────────────────────────────────────────────────────────────────────

/**
 * A single dimension: either a concrete non-negative integer or -1 (dynamic).
 * The value -1 means the dimension is unknown at graph-construction time.
 */
export type Dim = number;

/**
 * An ordered list of dimensions.  Compatible with the compiler's `Shape` type
 * (`number[]`) so values can be passed directly to `graph.addInputTensor()`.
 */
export type ShapeExpr = readonly Dim[];

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Return true when the two dimensions are broadcast-compatible.
 * Two dims d1 and d2 are compatible iff at least one is 1, both equal, or
 * either is -1 (dynamic, assumed compatible at compile time).
 */
function dimsCompatible(d1: Dim, d2: Dim): boolean {
  return d1 === -1 || d2 === -1 || d1 === 1 || d2 === 1 || d1 === d2;
}

/**
 * Compute the broadcast output dimension for a single axis.
 * Returns -1 when either input is dynamic.
 */
function broadcastDim(d1: Dim, d2: Dim): Dim {
  if (d1 === -1 || d2 === -1) return -1;
  if (d1 === 1) return d2;
  if (d2 === 1) return d1;
  return d1; // they must be equal (guaranteed by dimsCompatible check)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * NumPy-style broadcast shape for two tensors.
 *
 * Aligns shapes from the right, pads the shorter one with 1s on the left,
 * then checks each axis for compatibility.
 *
 * @throws {ShapeError} when the shapes are not broadcast-compatible.
 */
export function broadcast(a: ShapeExpr, b: ShapeExpr): ShapeExpr {
  const rank = Math.max(a.length, b.length);
  const aPadded = [...Array(rank - a.length).fill(1), ...a];
  const bPadded = [...Array(rank - b.length).fill(1), ...b];

  const result: Dim[] = [];
  for (let i = 0; i < rank; i++) {
    const da = aPadded[i];
    const db = bPadded[i];
    if (!dimsCompatible(da, db)) {
      throw new ShapeError(
        `Cannot broadcast shapes [${a.join(", ")}] and [${b.join(", ")}]: ` +
        `dimension ${i} has sizes ${da} and ${db}`,
        { shapeA: [...a], shapeB: [...b], axis: i },
      );
    }
    result.push(broadcastDim(da, db));
  }
  return result;
}

/**
 * Shape inference for matrix multiplication.
 *
 * Supports:
 *   - 2-D: [M, K] × [K, N] → [M, N]
 *   - Batched: [..., M, K] × [..., K, N] → [..., M, N]
 *     (batch dims are broadcast-compatible)
 *
 * @throws {ShapeError} on rank < 2 or K-dimension mismatch.
 */
export function matmulShape(a: ShapeExpr, b: ShapeExpr): ShapeExpr {
  if (a.length < 2 || b.length < 2) {
    throw new ShapeError(
      `matmul requires rank ≥ 2; got [${a.join(", ")}] and [${b.join(", ")}]`,
      { shapeA: [...a], shapeB: [...b] },
    );
  }

  const aM = a[a.length - 2];
  const aK = a[a.length - 1];
  const bK = b[b.length - 2];
  const bN = b[b.length - 1];

  // K must match (or one must be dynamic)
  if (aK !== -1 && bK !== -1 && aK !== bK) {
    throw new ShapeError(
      `matmul inner dimension mismatch: [${a.join(", ")}] × [${b.join(", ")}]`,
      { shapeA: [...a], shapeB: [...b] },
    );
  }

  const aBatch = a.slice(0, -2);
  const bBatch = b.slice(0, -2);

  const batchShape = aBatch.length === 0 && bBatch.length === 0
    ? []
    : broadcast(aBatch, bBatch);

  return [...batchShape, aM, bN];
}

/**
 * Shape after applying a permutation (transpose).
 *
 * @param shape  Input shape.
 * @param perm   Permutation of `[0, 1, ..., rank-1]`.
 * @throws {ShapeError} when the permutation is invalid.
 */
export function transposeShape(shape: ShapeExpr, perm: readonly number[]): ShapeExpr {
  if (perm.length !== shape.length) {
    throw new ShapeError(
      `transpose permutation length ${perm.length} does not match rank ${shape.length}`,
      { shape: [...shape], perm: [...perm] },
    );
  }
  const seen = new Set<number>();
  for (const p of perm) {
    if (p < 0 || p >= shape.length || !Number.isInteger(p)) {
      throw new ShapeError(
        `transpose permutation contains invalid index ${p} for rank ${shape.length}`,
        { shape: [...shape], perm: [...perm] },
      );
    }
    if (seen.has(p)) {
      throw new ShapeError(
        `transpose permutation contains duplicate index ${p}`,
        { shape: [...shape], perm: [...perm] },
      );
    }
    seen.add(p);
  }
  return perm.map(p => shape[p]);
}

/**
 * Shape after a reshape.  Exactly one dimension may be -1, in which case it
 * is inferred from the total element count.
 *
 * Returns the resolved output shape (with -1 replaced by the inferred value).
 * When the input shape contains dynamic dims (-1), inference is skipped and
 * the target shape is returned as-is.
 *
 * @throws {ShapeError} when the element count is incompatible or -1 appears
 *   more than once.
 */
export function reshapeShape(input: ShapeExpr, target: ShapeExpr): ShapeExpr {
  const wildIdx = target.findIndex(d => d === -1);
  const wildCount = target.filter(d => d === -1).length;

  if (wildCount > 1) {
    throw new ShapeError(
      `reshape target shape may contain at most one -1; got [${target.join(", ")}]`,
      { input: [...input], target: [...target] },
    );
  }

  // If input has any dynamic dimension, skip element-count validation
  if (input.includes(-1)) {
    return [...target];
  }

  const inElems  = input.reduce((a, d) => a * d, 1);
  const knownOut = target.reduce((a, d) => (d === -1 ? a : a * d), 1);

  if (wildIdx >= 0) {
    if (inElems % knownOut !== 0) {
      throw new ShapeError(
        `Cannot reshape [${input.join(", ")}] (${inElems} elems) into [${target.join(", ")}]`,
        { input: [...input], target: [...target] },
      );
    }
    const resolved = [...target] as Dim[];
    resolved[wildIdx] = inElems / knownOut;
    return resolved;
  }

  if (inElems !== knownOut) {
    throw new ShapeError(
      `Cannot reshape [${input.join(", ")}] (${inElems} elems) into [${target.join(", ")}] (${knownOut} elems)`,
      { input: [...input], target: [...target] },
    );
  }

  return [...target];
}

/**
 * Shape after a reduction (sum / mean) along the given axes.
 *
 * @param keepDims  When true, reduced axes are kept as size-1.
 * @throws {ShapeError} when any axis is out of range.
 */
export function reduceShape(
  input: ShapeExpr,
  axes: readonly number[],
  keepDims: boolean,
): ShapeExpr {
  const rank = input.length;
  // Normalise negative axes
  const normalised = axes.map(ax => (ax < 0 ? ax + rank : ax));

  for (const ax of normalised) {
    if (ax < 0 || ax >= rank) {
      throw new ShapeError(
        `reduce axis ${ax} is out of range for rank-${rank} tensor`,
        { input: [...input], axes: [...axes] },
      );
    }
  }

  const axisSet = new Set(normalised);
  const result: Dim[] = [];
  for (let i = 0; i < rank; i++) {
    if (axisSet.has(i)) {
      if (keepDims) result.push(1);
    } else {
      result.push(input[i]);
    }
  }
  return result;
}

/**
 * Return the total number of elements in a fully static shape.
 * Returns -1 when any dimension is dynamic.
 */
export function shapeNumel(shape: ShapeExpr): number {
  if (shape.includes(-1)) return -1;
  return shape.reduce((a, d) => a * d, 1);
}

/** Shallow equality check for two shapes. */
export function shapesEqual(a: ShapeExpr, b: ShapeExpr): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}
