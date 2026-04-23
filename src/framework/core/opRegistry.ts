// ─────────────────────────────────────────────────────────────────────────────
// frontend/core/opRegistry.ts
//
// Op schema registry: validates arity and infers output shapes/dtypes.
// ─────────────────────────────────────────────────────────────────────────────

import {
  ShapeExpr, broadcast, matmulShape, transposeShape, reshapeShape, reduceShape,
} from "./shape";
import { OpError, ShapeError } from "./errors";
import { IRDType, IRAttrs }    from "../ir/schema";

// ─── Core types ───────────────────────────────────────────────────────────────

export interface InputSpec  { readonly name: string; readonly dtype?: "any" | IRDType; }
export interface OutputSpec { readonly name: string; }

export interface InferContext {
  readonly inputShapes: readonly ShapeExpr[];
  readonly inputDtypes: readonly IRDType[];
  readonly attrs:       IRAttrs;
}

export type GradBuilderFn = (
  fwdCtx:    InferContext,
  outputIds: readonly string[],
  gradIds:   readonly string[],
  applyOp:   (op: string, inputs: readonly string[], attrs?: IRAttrs) => string[],
  /** bwd-side tensor ids for the forward input tensors (same order as node.inputs) */
  inputIds:  readonly string[],
) => string[];

export interface OpSchema {
  readonly op:      string;
  readonly inputs:  readonly InputSpec[];
  readonly outputs: readonly OutputSpec[];
  inferShape(ctx: InferContext): Array<{ shape: ShapeExpr; dtype: IRDType }>;
  validateAttrs?(attrs: IRAttrs): void;
  gradBuilder?: GradBuilderFn;
}

// ─── Registry ─────────────────────────────────────────────────────────────────

export class OpSchemaRegistry {
  private readonly _schemas: Map<string, OpSchema> = new Map();

  constructor(initial: readonly OpSchema[] = []) {
    for (const s of initial) this._schemas.set(s.op, s);
  }

  register(schema: OpSchema): this {
    this._schemas.set(schema.op, schema);
    return this;
  }

  get(op: string): OpSchema {
    const s = this._schemas.get(op);
    if (!s) throw new OpError(`Unknown op "${op}"`, { op });
    return s;
  }

  has(op: string): boolean { return this._schemas.has(op); }
  get opNames(): readonly string[] { return [...this._schemas.keys()]; }

  infer(
    op:          string,
    inputShapes: readonly ShapeExpr[],
    inputDtypes: readonly IRDType[],
    attrs:       IRAttrs,
  ): Array<{ name: string; shape: ShapeExpr; dtype: IRDType }> {
    const schema = this.get(op);
    if (inputShapes.length !== schema.inputs.length) {
      throw new OpError(
        `Op "${op}" expects ${schema.inputs.length} input(s), got ${inputShapes.length}`,
        { op, expected: schema.inputs.length, got: inputShapes.length },
      );
    }
    if (schema.validateAttrs) schema.validateAttrs(attrs);
    const ctx      = { inputShapes, inputDtypes, attrs };
    const inferred = schema.inferShape(ctx);
    return inferred.map((out, i) => ({
      name:  schema.outputs[i]?.name ?? `out${i}`,
      shape: out.shape,
      dtype: out.dtype,
    }));
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function dominantDtype(dtypes: readonly IRDType[]): IRDType {
  if (dtypes.includes("float64")) return "float64";
  if (dtypes.includes("float32")) return "float32";
  if (dtypes.includes("int64"))   return "int64";
  if (dtypes.includes("int32"))   return "int32";
  return dtypes[0] ?? "float32";
}

function elementwiseUnary(op: string, gradBuilder?: GradBuilderFn): OpSchema {
  return {
    op,
    inputs:  [{ name: "x" }],
    outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes }) {
      return [{ shape: [...inputShapes[0]], dtype: inputDtypes[0] }];
    },
    gradBuilder,
  };
}

function elementwiseBinary(op: string, gradBuilder?: GradBuilderFn): OpSchema {
  return {
    op,
    inputs:  [{ name: "lhs" }, { name: "rhs" }],
    outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes }) {
      const shape = broadcast(inputShapes[0], inputShapes[1]);
      return [{ shape, dtype: dominantDtype(inputDtypes) }];
    },
    gradBuilder,
  };
}

// ─── Default schemas ──────────────────────────────────────────────────────────

export const DEFAULT_OP_SCHEMAS: readonly OpSchema[] = [
  elementwiseBinary("add"),
  elementwiseBinary("sub"),
  elementwiseBinary("mul"),
  elementwiseBinary("div"),
  elementwiseUnary("relu"),
  elementwiseUnary("sigmoid"),
  elementwiseUnary("tanh"),
  elementwiseUnary("gelu"),
  elementwiseUnary("exp"),
  elementwiseUnary("sqrt"),
  elementwiseUnary("neg"),
  elementwiseUnary("abs"),
  {
    op: "matmul", inputs: [{ name: "a" }, { name: "b" }], outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes }) {
      return [{ shape: matmulShape(inputShapes[0], inputShapes[1]), dtype: dominantDtype(inputDtypes) }];
    },
  },
  {
    op: "reshape", inputs: [{ name: "x" }], outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const target = attrs["shape"] as number[] | undefined;
      if (!target || !Array.isArray(target)) {
        throw new OpError("reshape requires an attrs.shape array", { attrs });
      }
      return [{ shape: reshapeShape(inputShapes[0], target), dtype: inputDtypes[0] }];
    },
    validateAttrs(attrs) {
      if (!Array.isArray(attrs["shape"])) {
        throw new OpError("reshape attrs.shape must be an array of integers");
      }
    },
  },
  {
    op: "transpose", inputs: [{ name: "x" }], outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const rank = inputShapes[0].length;
      const perm = (attrs["perm"] as number[] | undefined) ??
        Array.from({ length: rank }, (_, i) => rank - 1 - i);
      return [{ shape: transposeShape(inputShapes[0], perm), dtype: inputDtypes[0] }];
    },
  },
  {
    op: "sum", inputs: [{ name: "x" }], outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const axes     = (attrs["axes"]     as number[] | undefined) ?? [];
      const keepDims = (attrs["keepDims"] as boolean  | undefined) ?? false;
      return [{ shape: reduceShape(inputShapes[0], axes, keepDims), dtype: inputDtypes[0] }];
    },
  },
  {
    op: "mean", inputs: [{ name: "x" }], outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const axes     = (attrs["axes"]     as number[] | undefined) ?? [];
      const keepDims = (attrs["keepDims"] as boolean  | undefined) ?? false;
      return [{ shape: reduceShape(inputShapes[0], axes, keepDims), dtype: inputDtypes[0] }];
    },
  },
  {
    // 2-D spatial pooling (max/avg).  Input: [N, H, W, C] or [N, C, H, W].
    // Output spatial dims are floor(dim / stride) each.
    op: "pool2d", inputs: [{ name: "x" }], outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const shape  = [...inputShapes[0]] as number[];
      const stride = (attrs["stride"] as number | undefined) ?? 2;
      // Shrink spatial dims (indices 1 and 2 for 4-D input).
      if (shape.length === 4) {
        shape[1] = Math.floor(shape[1] / stride);
        shape[2] = Math.floor(shape[2] / stride);
      }
      return [{ shape, dtype: inputDtypes[0] }];
    },
    validateAttrs(attrs) {
      const ks = attrs["kernelSize"];
      const st = attrs["stride"];
      if (ks !== undefined && typeof ks !== "number") {
        throw new OpError("pool2d attrs.kernelSize must be a number");
      }
      if (st !== undefined && typeof st !== "number") {
        throw new OpError("pool2d attrs.stride must be a number");
      }
    },
  },
];

export const defaultOpRegistry = new OpSchemaRegistry(DEFAULT_OP_SCHEMAS);
