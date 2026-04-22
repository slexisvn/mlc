// ─────────────────────────────────────────────────────────────────────────────
// framework/tensor.ts
//
// SymbolicTensor — a typed handle to a value flowing through a GraphBuilder.
//
// A SymbolicTensor is intentionally immutable from user-code perspective:
// users read its shape and dtype but never mutate it.  All graph mutations go
// through GraphBuilder.applyOp().
//
// The class is thin by design.  Rich operations are exposed through the
// functional API in ops.ts, and the Module class in module.ts.
// ─────────────────────────────────────────────────────────────────────────────

import { ShapeExpr } from "./shape";
import { IRDType } from "../shared-ir/schema";
import { TensorId } from "../shared-ir/ids";

/**
 * A symbolic reference to a tensor value in a computation graph.
 *
 * Instances are created by `GraphBuilder.input()`, `GraphBuilder.parameter()`,
 * and `GraphBuilder.applyOp()`.  They should not be constructed directly.
 */
export class SymbolicTensor {
  /** Stable id within the owning GraphBuilder (e.g. "t0", "t1", …). */
  readonly id:    TensorId;
  /** Human-readable name (e.g. "x", "fc1.weight"). */
  readonly name:  string;
  readonly dtype: IRDType;
  readonly shape: ShapeExpr;
  /** Whether this tensor holds a trainable parameter. */
  readonly isParam: boolean;

  /** @internal — constructed by GraphBuilder only. */
  constructor(
    id:      TensorId,
    name:    string,
    dtype:   IRDType,
    shape:   ShapeExpr,
    isParam: boolean = false,
  ) {
    this.id      = id;
    this.name    = name;
    this.dtype   = dtype;
    this.shape   = shape;
    this.isParam = isParam;
  }

  /** Scalar tensor (rank 0). */
  get isScalar(): boolean { return this.shape.length === 0; }

  /** Rank of the tensor (number of dimensions). */
  get rank(): number { return this.shape.length; }

  toString(): string {
    return `Tensor(${this.name}, shape=[${this.shape.join(", ")}], dtype=${this.dtype})`;
  }
}
