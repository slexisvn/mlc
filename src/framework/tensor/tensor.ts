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

  add(other: TensorLike): SymbolicTensor { return _applyBinary("add", this, other); }
  sub(other: TensorLike): SymbolicTensor { return _applyBinary("sub", this, other); }
  mul(other: TensorLike): SymbolicTensor { return _applyBinary("mul", this, other); }
  div(other: TensorLike): SymbolicTensor { return _applyBinary("div", this, other); }

  reshape(shape: number[]): SymbolicTensor {
    const gb = _getActiveBuilder();
    if (!gb) throw new Error("Cannot run reshape outside ExportSession build.");
    return gb.applyOp("reshape", [this], { shape } as unknown as Record<string, unknown>)[0];
  }

  transpose(perm?: number[]): SymbolicTensor {
    const gb    = _getActiveBuilder();
    if (!gb) throw new Error("Cannot run transpose outside ExportSession build.");
    const attrs = perm ? { perm } as unknown as Record<string, unknown> : {};
    return gb.applyOp("transpose", [this], attrs)[0];
  }

  matmul(other: TensorLike): SymbolicTensor { return _applyBinary("matmul", this, other); }

  /**
   * 2-D pooling op.  The compiler contract requires NCHW input;
   * LayoutInsertionPass will insert a transpose automatically when needed.
   */
  pool2d(kernelSize: number = 2, stride: number = 2): SymbolicTensor {
    const gb = _getActiveBuilder();
    if (!gb) throw new Error("Cannot run pool2d outside ExportSession build.");
    return gb.applyOp("pool2d", [this], { kernelSize, stride } as unknown as Record<string, unknown>)[0];
  }

  toString(): string {
    return `Tensor(${this.name}, shape=[${this.shape.join(", ")}], dtype=${this.dtype})`;
  }
}

export class EagerTensor {
  readonly isEager = true;
  constructor(
    public readonly data: number[],
    public readonly shape: number[],
    public readonly dtype: IRDType = "float32"
  ) {}

  get isScalar(): boolean { return this.shape.length === 0; }
  get rank():     number  { return this.shape.length; }

  toString(): string {
    return `EagerTensor(shape=[${this.shape.join(", ")}], dtype=${this.dtype})`;
  }

  private _noEager(): any { throw new Error("Eager execution is not supported. Use within an @compile traced layer."); }
  relu(): SymbolicTensor { return this._noEager(); }
  sigmoid(): SymbolicTensor { return this._noEager(); }
  tanh(): SymbolicTensor { return this._noEager(); }
  gelu(): SymbolicTensor { return this._noEager(); }
  softmax(): SymbolicTensor { return this._noEager(); }
  exp(): SymbolicTensor { return this._noEager(); }
  sqrt(): SymbolicTensor { return this._noEager(); }
  neg(): SymbolicTensor { return this._noEager(); }
  abs(): SymbolicTensor { return this._noEager(); }
  add(other: TensorLike): SymbolicTensor { return this._noEager(); }
  sub(other: TensorLike): SymbolicTensor { return this._noEager(); }
  mul(other: TensorLike): SymbolicTensor { return this._noEager(); }
  div(other: TensorLike): SymbolicTensor { return this._noEager(); }
  reshape(shape: number[]): SymbolicTensor { return this._noEager(); }
  transpose(perm?: number[]): SymbolicTensor { return this._noEager(); }
  matmul(other: TensorLike): SymbolicTensor { return this._noEager(); }
  pool2d(kernelSize?: number, stride?: number): SymbolicTensor { return this._noEager(); }
}

/** Public alias — matches the PyTorch mental model. */
export type Tensor = SymbolicTensor | EagerTensor;

export type TensorLike = Tensor | number | number[];

// ─── Internal helpers ─────────────────────────────────────────────────────────
// Lazy-requires core/context to avoid a circular static import chain:
//   graphBuilder → tensor → context → graphBuilder

/** Minimal structural type so tensor.ts does not import GraphBuilder directly. */
type _BuilderLike = {
  applyOp(op: string, inputs: readonly SymbolicTensor[], attrs?: Record<string, unknown>): SymbolicTensor[];
  param(name: string, dtype: IRDType, shape: ShapeExpr): SymbolicTensor;
  nodes: { size: number };
};

function _getActiveBuilder(): _BuilderLike | null {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ctx = require("../core/context") as { getActiveBuilder(): _BuilderLike | null, hasActiveContext(): boolean };
  return ctx.hasActiveContext() ? ctx.getActiveBuilder() : null;
}

function _promoteToSymbolic(t: TensorLike): SymbolicTensor {
  if (t instanceof SymbolicTensor) return t;
  
  const ctx = require("../core/context") as { getActiveParamSink(): any[] };
  const gb = _getActiveBuilder();
  if (!gb) {
    throw new Error("Cannot promote EagerTensor to SymbolicTensor outside of an active ExportSession build context.");
  }
  
  let eager: EagerTensor;
  if (t instanceof EagerTensor) {
    eager = t;
  } else if (typeof t === "number") {
    eager = new EagerTensor([t], []);
  } else {
    eager = new EagerTensor(t, [t.length]);
  }
  
  const name = `const_${gb.nodes.size}`;
  const tensor = gb.param(name, eager.dtype, eager.shape);
  ctx.getActiveParamSink().push({ tensor, data: eager.data, isConst: true });
  return tensor;
}

function _applyUnary(op: string, x: SymbolicTensor): SymbolicTensor {
  const gb = _getActiveBuilder();
  if (!gb) throw new Error(`Cannot run ${op} outside ExportSession build.`);
  return gb.applyOp(op, [x])[0];
}

function _applyBinary(op: string, a: SymbolicTensor, b: TensorLike): SymbolicTensor {
  const gb = _getActiveBuilder();
  if (!gb) throw new Error(`Cannot run ${op} outside ExportSession build.`);
  return gb.applyOp(op, [a, _promoteToSymbolic(b)])[0];
}
