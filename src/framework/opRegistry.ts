// ─────────────────────────────────────────────────────────────────────────────
// framework/opRegistry.ts
//
// Op schema registry for the framework frontend.
//
// Each OpSchema captures the static contract of an operator:
//   • How many inputs it accepts (arity check).
//   • How to infer the output shape and dtype from the inputs.
//   • Whether it is differentiable and, if so, a gradient builder function.
//   • Optional attribute validators.
//
// The registry is populated once at startup and queried by GraphBuilder for
// every `applyOp()` call.  It is extensible: user code can register custom
// ops via `opRegistry.register(schema)`.
//
// This file also exports the DEFAULT_OP_SCHEMAS array so the default ops are
// easy to enumerate in tests.
// ─────────────────────────────────────────────────────────────────────────────

import { ShapeExpr, broadcast, matmulShape, transposeShape, reduceShape } from "./shape";
import { OpError, ShapeError } from "./errors";
import { IRDType, IRAttrs } from "../shared-ir/schema";

// ─── Core types ───────────────────────────────────────────────────────────────

/**
 * Describes one input slot of an op.
 * Used for arity checking and error messages.
 */
export interface InputSpec {
  readonly name:     string;
  readonly dtype?:   "any" | IRDType;   // omit = any dtype accepted
}

/**
 * Describes one output slot of an op.
 */
export interface OutputSpec {
  readonly name: string;
}

/**
 * Context passed to `inferShape` and `gradBuilder`.
 * Contains the resolved shapes and dtypes of all input tensors.
 */
export interface InferContext {
  readonly inputShapes: readonly ShapeExpr[];
  readonly inputDtypes: readonly IRDType[];
  readonly attrs:       IRAttrs;
}

/**
 * A gradient builder receives the shapes/dtypes of the forward op's inputs
 * and outputs, plus the upstream gradient tensor id for each output, and
 * returns a list of downstream gradient tensor ids — one per forward input.
 *
 * The builder creates new nodes through the GraphBuilder by calling its
 * `applyOp` method and returns the resulting symbolic gradient tensor ids.
 * The `GradBuilderFn` type is intentionally kept generic here; the concrete
 * `SymbolicTensor` type from tensor.ts is avoided to prevent circular imports.
 * The actual gradient builders live in autodiff.ts.
 */
export type GradBuilderFn = (
  fwdCtx:    InferContext,
  outputIds: readonly string[],  // forward output tensor symbolic ids
  gradIds:   readonly string[],  // upstream gradient tensor symbolic ids
  applyOp:   (
    op:       string,
    inputs:   readonly string[],
    attrs?:   IRAttrs,
  ) => string[],                 // returns new output symbolic ids
) => string[];                   // downstream gradient ids, one per fwd input

/**
 * Complete schema for a single op.
 */
export interface OpSchema {
  /** Op name — must match the name used in the compiler's op registry. */
  readonly op: string;

  /** Ordered input slots.  Length determines accepted arity. */
  readonly inputs: readonly InputSpec[];

  /** Ordered output slots.  Most ops have exactly one output. */
  readonly outputs: readonly OutputSpec[];

  /**
   * Infer the output shapes and dtypes from input shapes/dtypes.
   * Must return exactly `outputs.length` entries.
   */
  inferShape(ctx: InferContext): Array<{ shape: ShapeExpr; dtype: IRDType }>;

  /** Validate attrs beyond basic type-checking.  Throw OpError on failure. */
  validateAttrs?(attrs: IRAttrs): void;

  /**
   * Optional gradient builder.  When absent the op is treated as
   * non-differentiable and any backward pass through it will throw.
   */
  gradBuilder?: GradBuilderFn;
}

// ─── Registry ─────────────────────────────────────────────────────────────────

export class OpSchemaRegistry {
  private readonly _schemas: Map<string, OpSchema> = new Map();

  constructor(initial: readonly OpSchema[] = []) {
    for (const s of initial) this._schemas.set(s.op, s);
  }

  /**
   * Register a new schema, or overwrite an existing one.
   * Returns `this` for fluent chaining.
   */
  register(schema: OpSchema): this {
    this._schemas.set(schema.op, schema);
    return this;
  }

  /**
   * Look up a schema by op name.
   * @throws {OpError} when the op is not registered.
   */
  get(op: string): OpSchema {
    const s = this._schemas.get(op);
    if (!s) throw new OpError(`Unknown op "${op}"`, { op });
    return s;
  }

  has(op: string): boolean { return this._schemas.has(op); }

  /** All registered op names. */
  get opNames(): readonly string[] { return [...this._schemas.keys()]; }

  /**
   * Validate inputs against a schema and return the inferred output descriptors.
   *
   * @throws {OpError}   on arity mismatch.
   * @throws {ShapeError} on shape/dtype inference failure.
   */
  infer(
    op: string,
    inputShapes: readonly ShapeExpr[],
    inputDtypes: readonly IRDType[],
    attrs: IRAttrs,
  ): Array<{ name: string; shape: ShapeExpr; dtype: IRDType }> {
    const schema = this.get(op);

    if (inputShapes.length !== schema.inputs.length) {
      throw new OpError(
        `Op "${op}" expects ${schema.inputs.length} input(s), got ${inputShapes.length}`,
        { op, expected: schema.inputs.length, got: inputShapes.length },
      );
    }

    if (schema.validateAttrs) schema.validateAttrs(attrs);

    const ctx: InferContext = { inputShapes, inputDtypes, attrs };
    const inferred = schema.inferShape(ctx);

    return inferred.map((out, i) => ({
      name:  schema.outputs[i]?.name ?? `out${i}`,
      shape: out.shape,
      dtype: out.dtype,
    }));
  }
}

// ─── Helper: dominant dtype ───────────────────────────────────────────────────

function dominantDtype(dtypes: readonly IRDType[]): IRDType {
  // Simple rule: float64 > float32 > int64 > int32 > bool
  if (dtypes.includes("float64")) return "float64";
  if (dtypes.includes("float32")) return "float32";
  if (dtypes.includes("int64"))   return "int64";
  if (dtypes.includes("int32"))   return "int32";
  return dtypes[0] ?? "float32";
}

// ─── Helper: elementwise schema ───────────────────────────────────────────────

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

// ─── Default op schemas ───────────────────────────────────────────────────────

export const DEFAULT_OP_SCHEMAS: readonly OpSchema[] = [

  // ── Binary elementwise ───────────────────────────────────────────────────
  elementwiseBinary("add"),
  elementwiseBinary("sub"),
  elementwiseBinary("mul"),
  elementwiseBinary("div"),

  // ── Unary elementwise ────────────────────────────────────────────────────
  elementwiseUnary("relu"),
  elementwiseUnary("sigmoid"),
  elementwiseUnary("tanh"),
  elementwiseUnary("gelu"),
  elementwiseUnary("exp"),
  elementwiseUnary("sqrt"),
  elementwiseUnary("neg"),
  elementwiseUnary("abs"),

  // ── matmul ───────────────────────────────────────────────────────────────
  {
    op:      "matmul",
    inputs:  [{ name: "a" }, { name: "b" }],
    outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes }) {
      const shape = matmulShape(inputShapes[0], inputShapes[1]);
      return [{ shape, dtype: dominantDtype(inputDtypes) }];
    },
  },

  // ── reshape ──────────────────────────────────────────────────────────────
  {
    op:      "reshape",
    inputs:  [{ name: "x" }],
    outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const target = attrs["shape"] as number[] | undefined;
      if (!target || !Array.isArray(target)) {
        throw new OpError('reshape requires an attrs.shape array', { attrs });
      }
      // Import lazily to avoid circular dependency chain through shape.ts
      const { reshapeShape } = require("./shape") as typeof import("./shape");
      const out = reshapeShape(inputShapes[0], target);
      return [{ shape: out, dtype: inputDtypes[0] }];
    },
    validateAttrs(attrs) {
      if (!Array.isArray(attrs["shape"])) {
        throw new OpError('reshape attrs.shape must be an array of integers');
      }
    },
  },

  // ── transpose ────────────────────────────────────────────────────────────
  {
    op:      "transpose",
    inputs:  [{ name: "x" }],
    outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const perm = attrs["perm"] as number[] | undefined;
      if (!perm) {
        // Default: reverse dims
        const rank = inputShapes[0].length;
        const defaultPerm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
        const shape = transposeShape(inputShapes[0], defaultPerm);
        return [{ shape, dtype: inputDtypes[0] }];
      }
      const shape = transposeShape(inputShapes[0], perm);
      return [{ shape, dtype: inputDtypes[0] }];
    },
  },

  // ── sum ──────────────────────────────────────────────────────────────────
  {
    op:      "sum",
    inputs:  [{ name: "x" }],
    outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const axes     = (attrs["axes"]     as number[] | undefined) ?? [];
      const keepDims = (attrs["keepDims"] as boolean  | undefined) ?? false;
      const shape    = reduceShape(inputShapes[0], axes, keepDims);
      return [{ shape, dtype: inputDtypes[0] }];
    },
  },

  // ── mean ─────────────────────────────────────────────────────────────────
  {
    op:      "mean",
    inputs:  [{ name: "x" }],
    outputs: [{ name: "y" }],
    inferShape({ inputShapes, inputDtypes, attrs }) {
      const axes     = (attrs["axes"]     as number[] | undefined) ?? [];
      const keepDims = (attrs["keepDims"] as boolean  | undefined) ?? false;
      const shape    = reduceShape(inputShapes[0], axes, keepDims);
      return [{ shape, dtype: inputDtypes[0] }];
    },
  },
];

// ─── Singleton default registry ───────────────────────────────────────────────

/** Pre-populated with all default ops.  Import and extend as needed. */
export const defaultOpRegistry = new OpSchemaRegistry(DEFAULT_OP_SCHEMAS);
