// ─────────────────────────────────────────────────────────────────────────────
//
// ExportSession — owns a single computation graph and drives model tracing.
//
// This is the primary entry point for defining models with the PyTorch-like API.
//
// Typical usage:
// ```ts
// import * as nn from "./frontend/nn";
// import * as F  from "./frontend/functional";
// import { ExportSession, Tensor } from "./frontend/export";
//
// class MyModel extends nn.Module {
//   private fc1 = this.register("fc1", new nn.Linear(784, 256));
//   private fc2 = this.register("fc2", new nn.Linear(256, 10));
//
//   forward(x: Tensor): Tensor {
//     return F.relu(this.fc1.forward(x))
//             .pipe(h => this.fc2.forward(h));  // or just two lines
//   }
// }
//
// const session = new ExportSession({ id: "my_model" });
// session.build(ctx => {
//   const x = ctx.input("x", "float32", [32, 784]);
//   const model = new MyModel();
//   const logits = model.forward(x);
//   ctx.markOutput(logits);
// });
// const pkg = session.export();
// ```
// ─────────────────────────────────────────────────────────────────────────────

import { GraphBuilder, GraphBuilderOptions } from "../core/graphBuilder";
import { withActiveContext, ParamSpec }      from "../core/context";
import { SymbolicTensor }                    from "../tensor/tensor";
import { IRDType, IRPackage, ParameterData, GraphKind } from "../ir/schema";
import { ShapeExpr }                         from "../core/shape";
import { Initialiser, initZeros }            from "../nn/parameter";

// ─── SessionContext ───────────────────────────────────────────────────────────

/**
 * Scoped context object passed to the `build()` callback.
 * Used to declare graph inputs, parameters, and outputs.
 */
export class SessionContext {
  /** @internal */
  constructor(
    private readonly _gb:        GraphBuilder,
    private readonly _paramSink: ParamSpec[],
  ) {}

  /**
   * Declare a graph-level data input (runtime tensor, not a trainable param).
   */
  input(
    name:  string,
    dtype: IRDType   = "float32",
    shape: ShapeExpr = [],
  ): SymbolicTensor {
    return this._gb.input(name, dtype, shape);
  }

  /**
   * Declare a trainable parameter with initial data.
   * Useful for functional-style graph authoring without nn.Module.
   */
  param(
    name:  string,
    dtype: IRDType    = "float32",
    shape: ShapeExpr  = [],
    init:  Initialiser = initZeros,
  ): SymbolicTensor {
    const tensor = this._gb.param(name, dtype, shape);
    this._paramSink.push({ tensor, data: init(shape) });
    return tensor;
  }

  /**
   * Declare a compile-time constant tensor.
   * The provided values are embedded in the IRPackage and the bridge attaches a
   * `constantPayload` to the corresponding compiler tensor so ConstantFoldingPass
   * can evaluate downstream ops at compile time.
   */
  const(
    name:   string,
    dtype:  IRDType   = "float32",
    shape:  ShapeExpr = [],
    values: number[]  = [],
  ): SymbolicTensor {
    const tensor = this._gb.param(name, dtype, shape);
    this._paramSink.push({ tensor, data: values, isConst: true });
    return tensor;
  }

  /**
   * Mark one or more tensors as graph outputs.
   */
  markOutput(...tensors: SymbolicTensor[]): void {
    this._gb.markOutputs(...tensors);
  }
}

// ─── ExportSession ────────────────────────────────────────────────────────────

export interface ExportSessionOptions {
  /** Graph id — appears in the serialised IRPackage. Defaults to "main". */
  id?:           string;
  /** Op registry override. Uses the built-in default when omitted. */
  opRegistry?:   GraphBuilderOptions["opRegistry"];
  /** Default dtype for inputs/params. Defaults to "float32". */
  defaultDtype?: IRDType;
}

/**
 * Owns a single computation graph and drives model tracing.
 *
 * Call `build(ctx => { ... })` to enter the active graph context.  Inside the
 * callback, `nn.Module` constructors and all `F.*` functional ops operate on
 * this session's graph automatically.
 *
 * Call `export()` when done to obtain a serialisable `IRPackage`.
 */
export class ExportSession {
  private readonly _gb:         GraphBuilder;
  private readonly _paramSpecs: ParamSpec[] = [];

  constructor(options: ExportSessionOptions = {}) {
    this._gb = new GraphBuilder({
      id:           options.id ?? "main",
      opRegistry:   options.opRegistry,
      defaultDtype: options.defaultDtype,
    });
  }

  /**
   * Enter the active graph context and run the builder callback.
   *
   * The callback receives a `SessionContext` for declaring inputs/outputs.
   * All `nn.Module.forward()` calls and `F.*` functional ops inside the
   * callback are automatically wired into this session's graph.
   *
   * Returns `this` for fluent chaining.
   */
  build(fn: (ctx: SessionContext) => void): this {
    withActiveContext(this._gb, this._paramSpecs, () => {
      fn(new SessionContext(this._gb, this._paramSpecs));
    });
    return this;
  }

  /**
   * Finalise the graph and return a plain-data `IRPackage`.
   *
   * Seals the internal `GraphBuilder`.  Subsequent calls to `build()` will
   * throw a `GraphBuildError`.
   */
  export(kind: GraphKind = "forward"): IRPackage {
    const graphIR = this._gb.build(kind);

    const parameters: ParameterData[] = this._paramSpecs.map(spec => ({
      tensorId: spec.tensor.id,
      name:     spec.tensor.name,
      data:     Array.from(spec.data),
      dtype:    spec.tensor.dtype,
      shape:    [...spec.tensor.shape],
      ...(spec.isConst ? { isConst: true } : {}),
    }));

    return {
      irVersion:    "0.1",
      opsetVersion: "mini-ts-0.1",
      graphs:       [graphIR],
      parameters:   parameters.length > 0 ? parameters : undefined,
    };
  }

  /**
   * Low-level escape hatch: access the underlying `GraphBuilder` directly.
   * Prefer `build()` for normal model authoring.
   */
  get graphBuilder(): GraphBuilder { return this._gb; }
}
