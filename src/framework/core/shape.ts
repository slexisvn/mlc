// ─────────────────────────────────────────────────────────────────────────────
// frontend/core/shape.ts
//
// Symbolic and concrete shape utilities.
// ─────────────────────────────────────────────────────────────────────────────

import { ShapeError } from "./errors";

export type Dim       = number;
export type ShapeExpr = readonly Dim[];

function dimsCompatible(d1: Dim, d2: Dim): boolean {
  return d1 === -1 || d2 === -1 || d1 === 1 || d2 === 1 || d1 === d2;
}

function broadcastDim(d1: Dim, d2: Dim): Dim {
  if (d1 === -1 || d2 === -1) return -1;
  if (d1 === 1) return d2;
  if (d2 === 1) return d1;
  return d1;
}

export function broadcast(a: ShapeExpr, b: ShapeExpr): ShapeExpr {
  const rank    = Math.max(a.length, b.length);
  const aPadded = [...Array(rank - a.length).fill(1), ...a];
  const bPadded = [...Array(rank - b.length).fill(1), ...b];
  const result: Dim[] = [];
  for (let i = 0; i < rank; i++) {
    const da = aPadded[i], db = bPadded[i];
    if (!dimsCompatible(da, db)) {
      throw new ShapeError(
        `Cannot broadcast shapes [${a.join(",")}] and [${b.join(",")}]: dimension ${i} has sizes ${da} and ${db}`,
        { shapeA: [...a], shapeB: [...b], axis: i },
      );
    }
    result.push(broadcastDim(da, db));
  }
  return result;
}

export function matmulShape(a: ShapeExpr, b: ShapeExpr): ShapeExpr {
  if (a.length < 2 || b.length < 2) {
    throw new ShapeError(
      `matmul requires rank ≥ 2; got [${a.join(",")}] and [${b.join(",")}]`,
      { shapeA: [...a], shapeB: [...b] },
    );
  }
  const aK = a[a.length - 1], bK = b[b.length - 2];
  if (aK !== -1 && bK !== -1 && aK !== bK) {
    throw new ShapeError(
      `matmul inner dimension mismatch: [${a.join(",")}] × [${b.join(",")}]`,
      { shapeA: [...a], shapeB: [...b] },
    );
  }
  const aBatch = a.slice(0, -2), bBatch = b.slice(0, -2);
  const batchShape = aBatch.length === 0 && bBatch.length === 0
    ? [] : broadcast(aBatch, bBatch);
  return [...batchShape, a[a.length - 2], b[b.length - 1]];
}

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
      throw new ShapeError(`transpose: invalid index ${p}`, { shape: [...shape], perm: [...perm] });
    }
    if (seen.has(p)) {
      throw new ShapeError(`transpose: duplicate index ${p}`, { shape: [...shape], perm: [...perm] });
    }
    seen.add(p);
  }
  return perm.map(p => shape[p]);
}

export function reshapeShape(input: ShapeExpr, target: ShapeExpr): ShapeExpr {
  const wildIdx   = target.findIndex(d => d === -1);
  const wildCount = target.filter(d => d === -1).length;
  if (wildCount > 1) {
    throw new ShapeError(
      `reshape target may contain at most one -1; got [${target.join(",")}]`,
      { input: [...input], target: [...target] },
    );
  }
  if (input.includes(-1)) return [...target];
  const inElems  = input.reduce((a, d) => a * d, 1);
  const knownOut = target.reduce((a, d) => (d === -1 ? a : a * d), 1);
  if (wildIdx >= 0) {
    if (inElems % knownOut !== 0) {
      throw new ShapeError(
        `Cannot reshape [${input.join(",")}] (${inElems}) into [${target.join(",")}]`,
        { input: [...input], target: [...target] },
      );
    }
    const resolved = [...target] as Dim[];
    resolved[wildIdx] = inElems / knownOut;
    return resolved;
  }
  if (inElems !== knownOut) {
    throw new ShapeError(
      `Cannot reshape [${input.join(",")}] (${inElems}) into [${target.join(",")}] (${knownOut})`,
      { input: [...input], target: [...target] },
    );
  }
  return [...target];
}

export function reduceShape(
  input:    ShapeExpr,
  axes:     readonly number[],
  keepDims: boolean,
): ShapeExpr {
  const rank       = input.length;
  const normalised = axes.map(ax => (ax < 0 ? ax + rank : ax));
  for (const ax of normalised) {
    if (ax < 0 || ax >= rank) {
      throw new ShapeError(
        `reduce axis ${ax} out of range for rank-${rank} tensor`,
        { input: [...input], axes: [...axes] },
      );
    }
  }
  const axisSet = new Set(normalised);
  const result: Dim[] = [];
  for (let i = 0; i < rank; i++) {
    if (axisSet.has(i)) { if (keepDims) result.push(1); }
    else                { result.push(input[i]); }
  }
  return result;
}

export function shapeNumel(shape: ShapeExpr): number {
  if (shape.includes(-1)) return -1;
  return shape.reduce((a, d) => a * d, 1);
}

export function shapesEqual(a: ShapeExpr, b: ShapeExpr): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}
