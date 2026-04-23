import { ShapeExpr } from "../core/shape";
import { IRDType }   from "../ir/schema";
import { TensorId }  from "../ir/ids";

/**
 * A symbolic reference to a tensor value in a computation graph.
 *
 * Instances are created by `GraphBuilder.input()`, `GraphBuilder.param()`,
 * and `GraphBuilder.applyOp()`.  Do not construct directly.
 *
 * Instance helpers are available for common ops and delegate into the active
 * graph context (set by `ExportSession.build()`).
 */
export class SymbolicTensor {
  readonly id:      TensorId;
  readonly name:    string;
  readonly dtype:   IRDType;
  readonly shape:   ShapeExpr;
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

  get isScalar(): boolean { return this.shape.length === 0; }
  get rank():     number  { return this.shape.length; }

  relu():     SymbolicTensor { return _applyUnary("relu",    this); }
  sigmoid():  SymbolicTensor { return _applyUnary("sigmoid", this); }
  tanh():     SymbolicTensor { return _applyUnary("tanh",    this); }
  gelu():     SymbolicTensor { return _applyUnary("gelu",    this); }
  softmax():  SymbolicTensor { return _applyUnary("softmax", this); }
  exp():      SymbolicTensor { return _applyUnary("exp",     this); }
  sqrt():     SymbolicTensor { return _applyUnary("sqrt",    this); }
  neg():      SymbolicTensor { return _applyUnary("neg",     this); }
  abs():      SymbolicTensor { return _applyUnary("abs",     this); }

  add(other: SymbolicTensor): SymbolicTensor { return _applyBinary("add", this, other); }
  sub(other: SymbolicTensor): SymbolicTensor { return _applyBinary("sub", this, other); }
  mul(other: SymbolicTensor): SymbolicTensor { return _applyBinary("mul", this, other); }
  div(other: SymbolicTensor): SymbolicTensor { return _applyBinary("div", this, other); }

  reshape(shape: number[]): SymbolicTensor {
    const gb = _getActiveBuilder();
    return gb.applyOp("reshape", [this], { shape } as unknown as Record<string, unknown>)[0];
  }

  transpose(perm?: number[]): SymbolicTensor {
    const gb    = _getActiveBuilder();
    const attrs = perm ? { perm } as unknown as Record<string, unknown> : {};
    return gb.applyOp("transpose", [this], attrs)[0];
  }

  matmul(other: SymbolicTensor): SymbolicTensor { return _applyBinary("matmul", this, other); }

  /**
   * 2-D pooling op.  The compiler contract requires NCHW input;
   * LayoutInsertionPass will insert a transpose automatically when needed.
   */
  pool2d(kernelSize: number = 2, stride: number = 2): SymbolicTensor {
    const gb = _getActiveBuilder();
    return gb.applyOp("pool2d", [this], { kernelSize, stride } as unknown as Record<string, unknown>)[0];
  }

  toString(): string {
    return `Tensor(${this.name}, shape=[${this.shape.join(", ")}], dtype=${this.dtype})`;
  }
}

/** Public alias — matches the PyTorch mental model. */
export type Tensor = SymbolicTensor;

// ─── Internal helpers ─────────────────────────────────────────────────────────
// Lazy-requires core/context to avoid a circular static import chain:
//   graphBuilder → tensor → context → graphBuilder

/** Minimal structural type so tensor.ts does not import GraphBuilder directly. */
type _BuilderLike = {
  applyOp(op: string, inputs: readonly SymbolicTensor[], attrs?: Record<string, unknown>): SymbolicTensor[];
};

function _getActiveBuilder(): _BuilderLike {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ctx = require("../core/context") as { getActiveBuilder(): _BuilderLike };
  return ctx.getActiveBuilder();
}

function _applyUnary(op: string, x: SymbolicTensor): SymbolicTensor {
  return _getActiveBuilder().applyOp(op, [x])[0];
}

function _applyBinary(op: string, a: SymbolicTensor, b: SymbolicTensor): SymbolicTensor {
  return _getActiveBuilder().applyOp(op, [a, b])[0];
}
