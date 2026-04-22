// ─────────────────────────────────────────────────────────────────────────────
// framework/module.ts
//
// Module — base class for all neural network components.
//
// A Module represents a reusable sub-computation with trainable parameters.
// Subclasses override `forward()` to describe their computation using the
// functional op API (ops.ts) or direct GraphBuilder.applyOp() calls.
//
// The Module class handles:
//   • Automatic parameter tracking via a ParameterStore.
//   • Nested module composition (sub-modules register their params upward).
//   • A clean forward() interface that abstracts GraphBuilder ownership.
//
// Design notes:
//   • Modules are stateless with respect to graph construction: forward()
//     can be called multiple times (e.g. once for forward graph, once for
//     a weight-sharing branch) and will reuse the same parameter tensors.
//   • Parameter initialisation happens once in the constructor / init()
//     call, not on every forward().
//   • There is no eager execution in v1: forward() returns SymbolicTensors,
//     not numeric values.
// ─────────────────────────────────────────────────────────────────────────────

import { GraphBuilder }  from "./graphBuilder";
import { SymbolicTensor } from "./tensor";
import { ParameterStore, ParameterSpec, Initialiser, initXavier, initZeros } from "./parameter";
import { IRDType }        from "../shared-ir/schema";
import { ShapeExpr }      from "./shape";

// ─── Module base class ────────────────────────────────────────────────────────

/**
 * Base class for all neural network modules.
 *
 * Subclasses must implement `forward()`.
 *
 * Example:
 * ```ts
 * class Linear extends Module {
 *   private w: SymbolicTensor;
 *   private b: SymbolicTensor;
 *
 *   constructor(inFeatures: number, outFeatures: number, gb: GraphBuilder, store: ParameterStore) {
 *     super(gb, store);
 *     this.w = this.addParam("weight", [inFeatures, outFeatures], initXavier);
 *     this.b = this.addParam("bias",   [outFeatures],             initZeros);
 *   }
 *
 *   forward(x: SymbolicTensor): SymbolicTensor {
 *     const [mm]  = this.gb.applyOp("matmul", [x, this.w]);
 *     const [out] = this.gb.applyOp("add",    [mm, this.b]);
 *     return out;
 *   }
 * }
 * ```
 */
export abstract class Module {
  protected readonly gb:    GraphBuilder;
  protected readonly store: ParameterStore;

  constructor(gb: GraphBuilder, store: ParameterStore) {
    this.gb    = gb;
    this.store = store;
  }

  /**
   * Compute the forward pass.  Inputs and outputs are SymbolicTensors.
   * Subclasses define the concrete signature.
   */
  abstract forward(...inputs: SymbolicTensor[]): SymbolicTensor | SymbolicTensor[];

  // ─── Parameter registration ────────────────────────────────────────────────

  /**
   * Create and register a trainable parameter.
   *
   * @param name        Human-readable name, e.g. "weight", "bias".
   * @param shape       Tensor shape.
   * @param init        Initialiser function.  Defaults to Xavier uniform.
   * @param dtype       Element dtype.  Defaults to "float32".
   */
  protected addParam(
    name:  string,
    shape: ShapeExpr,
    init:  Initialiser = initXavier,
    dtype: IRDType = "float32",
  ): SymbolicTensor {
    const tensor = this.gb.param(name, dtype, shape);
    const spec: ParameterSpec = { tensor, data: init(shape) };
    this.store.add(spec);
    return tensor;
  }

  // ─── Sub-module composition ────────────────────────────────────────────────

  /**
   * Register a child module.  This is a no-op in the current implementation
   * because all modules share the same GraphBuilder and ParameterStore passed
   * at construction time.
   *
   * The method exists as a hook for future introspection / serialization
   * features (e.g. named sub-module enumeration).
   */
  protected addModule<T extends Module>(
    _name:   string,
    child:   T,
  ): T {
    return child;
  }
}

// ─── Built-in modules ─────────────────────────────────────────────────────────

/**
 * Fully-connected linear layer: y = x @ weight.T + bias.
 *
 * weight shape: [outFeatures, inFeatures]
 * bias shape:   [outFeatures]
 *
 * This uses the convention weight @ x^T = (x @ weight^T) so that x can be
 * a batch of row vectors with shape [batch, inFeatures].
 */
export class Linear extends Module {
  private readonly _weight: SymbolicTensor;
  private readonly _bias:   SymbolicTensor;
  readonly inFeatures:  number;
  readonly outFeatures: number;

  constructor(
    inFeatures:  number,
    outFeatures: number,
    gb:          GraphBuilder,
    store:       ParameterStore,
    opts: {
      bias?:   boolean;
      dtype?:  IRDType;
      prefix?: string;
    } = {},
  ) {
    super(gb, store);
    this.inFeatures  = inFeatures;
    this.outFeatures = outFeatures;

    const prefix = opts.prefix ? `${opts.prefix}.` : "";
    const dtype  = opts.dtype ?? "float32";

    this._weight = this.addParam(`${prefix}weight`, [inFeatures, outFeatures], initXavier, dtype);
    this._bias   = this.addParam(`${prefix}bias`,   [outFeatures],             initZeros,  dtype);
  }

  forward(x: SymbolicTensor): SymbolicTensor {
    const [mm]  = this.gb.applyOp("matmul", [x, this._weight]);
    const [out] = this.gb.applyOp("add",    [mm, this._bias]);
    return out;
  }

  get weight(): SymbolicTensor { return this._weight; }
  get bias():   SymbolicTensor { return this._bias;   }
}

/**
 * Multi-layer perceptron with configurable depth and activation.
 *
 * Stacks `hiddenSizes.length` Linear layers, each followed by the given
 * activation, then a final Linear output layer.
 */
export class MLP extends Module {
  private readonly _layers: Linear[];
  private readonly _activation: string;

  constructor(
    inputSize:   number,
    hiddenSizes: readonly number[],
    outputSize:  number,
    gb:          GraphBuilder,
    store:       ParameterStore,
    opts: {
      activation?: "relu" | "sigmoid" | "tanh" | "gelu";
      dtype?:      IRDType;
    } = {},
  ) {
    super(gb, store);
    this._activation = opts.activation ?? "relu";

    const sizes  = [inputSize, ...hiddenSizes, outputSize];
    this._layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const layer = new Linear(sizes[i], sizes[i + 1], gb, store, {
        dtype:  opts.dtype,
        prefix: `layer${i}`,
      });
      this._layers.push(layer);
    }
  }

  forward(x: SymbolicTensor): SymbolicTensor {
    let h = x;
    for (let i = 0; i < this._layers.length - 1; i++) {
      h = this._layers[i].forward(h);
      [h] = this.gb.applyOp(this._activation, [h]);
    }
    // Final layer: no activation
    h = this._layers[this._layers.length - 1].forward(h);
    return h;
  }

  get layers(): readonly Linear[] { return this._layers; }
}
