// ─────────────────────────────────────────────────────────────────────────────
// framework/parameter.ts
//
// Helpers for creating trainable parameters and graph inputs with initial data.
//
// The framework distinguishes three kinds of tensors:
//
//   Input      — runtime-supplied data (e.g. a mini-batch).
//                Created by GraphBuilder.input().
//
//   Parameter  — trainable weights stored as part of the model.
//                Created by GraphBuilder.param(), with an optional initialiser.
//                The initialiser's data is packaged into IRPackage.parameters
//                by the serializer.
//
// This file provides:
//   • ParameterSpec     — records the initial data alongside a SymbolicTensor.
//   • ParameterStore    — collects all parameters from a GraphBuilder build.
//   • initXavier / initZeros / initOnes — common weight initialisers.
// ─────────────────────────────────────────────────────────────────────────────

import { SymbolicTensor }                     from "./tensor";
import { ParameterData }                      from "../shared-ir/schema";
import { IRDType, IRShape }                   from "../shared-ir/schema";
import { TensorId }                           from "../shared-ir/ids";
import { ShapeExpr, shapeNumel }              from "./shape";

// ─── Initialiser type ─────────────────────────────────────────────────────────

/**
 * A weight initialiser function.
 * Receives the tensor shape and returns a flat row-major array of values.
 */
export type Initialiser = (shape: ShapeExpr) => number[];

// ─── Built-in initialisers ────────────────────────────────────────────────────

/**
 * Glorot / Xavier uniform initialiser.
 * Samples from U[-limit, +limit] where limit = sqrt(6 / (fan_in + fan_out)).
 *
 * For 1-D tensors (bias), fan_in = fan_out = shape[0] / 2.
 */
export function initXavier(shape: ShapeExpr): number[] {
  const n = shapeNumel(shape);
  if (n <= 0) return [];

  const fanIn  = shape.length >= 2 ? shape[shape.length - 2] : shape[0];
  const fanOut = shape[shape.length - 1] ?? fanIn;
  const limit  = Math.sqrt(6 / (fanIn + fanOut));

  return Array.from({ length: n }, () => (Math.random() * 2 - 1) * limit);
}

/** Initialise all elements to zero. */
export function initZeros(shape: ShapeExpr): number[] {
  const n = shapeNumel(shape);
  return Array(n < 0 ? 0 : n).fill(0);
}

/** Initialise all elements to one. */
export function initOnes(shape: ShapeExpr): number[] {
  const n = shapeNumel(shape);
  return Array(n < 0 ? 0 : n).fill(1);
}

/** Initialise all elements to a constant value. */
export function initConstant(value: number): Initialiser {
  return (shape: ShapeExpr) => {
    const n = shapeNumel(shape);
    return Array(n < 0 ? 0 : n).fill(value);
  };
}

// ─── ParameterSpec ────────────────────────────────────────────────────────────

/**
 * Associates a SymbolicTensor (the parameter handle in the graph) with its
 * initial data (produced by an Initialiser at model-construction time).
 */
export interface ParameterSpec {
  readonly tensor: SymbolicTensor;
  /** Flat row-major initial values.  May be empty for dynamic-shape params. */
  readonly data:   readonly number[];
}

// ─── ParameterStore ───────────────────────────────────────────────────────────

/**
 * Collects ParameterSpecs and converts them to the IRPackage format.
 *
 * Modules push their parameters here during `forward()`.  The serializer then
 * calls `toParameterData()` to extract the serialisable form.
 */
export class ParameterStore {
  private readonly _specs: ParameterSpec[] = [];
  private readonly _seen:  Set<TensorId>   = new Set();

  /**
   * Register a parameter.  Duplicate ids are silently ignored (useful when
   * `forward()` is called multiple times on the same module).
   */
  add(spec: ParameterSpec): void {
    if (this._seen.has(spec.tensor.id)) return;
    this._seen.add(spec.tensor.id);
    this._specs.push(spec);
  }

  /** Convert to the serialisable format consumed by IRPackage. */
  toParameterData(): ParameterData[] {
    return this._specs.map(spec => ({
      tensorId: spec.tensor.id,
      name:     spec.tensor.name,
      dtype:    spec.tensor.dtype as IRDType,
      shape:    [...spec.tensor.shape] as IRShape,
      data:     [...spec.data],
    }));
  }

  get size(): number { return this._specs.length; }
}
