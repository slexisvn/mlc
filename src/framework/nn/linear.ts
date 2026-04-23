// ─────────────────────────────────────────────────────────────────────────────
// frontend/nn/linear.ts
//
// Linear — fully-connected layer: y = x @ weight + bias
// ─────────────────────────────────────────────────────────────────────────────

import { Module }         from "./module";
import { initXavier, initZeros, Initialiser } from "./parameter";
import { SymbolicTensor } from "../tensor/tensor";
import { getActiveBuilder } from "../core/context";
import { IRDType }        from "../ir/schema";

export interface LinearOptions {
  /** Include a bias term. Defaults to true. */
  bias?:  boolean;
  dtype?: IRDType;
  weightInit?: Initialiser;
  biasInit?:   Initialiser;
}

/**
 * Fully-connected linear layer: y = x @ weight + bias.
 *
 * weight shape: [inFeatures, outFeatures]
 * bias shape:   [outFeatures]
 *
 * Example:
 * ```ts
 * class MyModel extends nn.Module {
 *   private fc = this.register("fc", new nn.Linear(784, 256));
 *   forward(x: Tensor) { return this.fc.forward(x).relu(); }
 * }
 * ```
 */
export class Linear extends Module {
  readonly inFeatures:  number;
  readonly outFeatures: number;

  private readonly _useBias:    boolean;
  private readonly _dtype:      IRDType;
  private readonly _weightInit: Initialiser;
  private readonly _biasInit:   Initialiser;

  private _weight: SymbolicTensor | null = null;
  private _bias:   SymbolicTensor | null = null;

  constructor(
    inFeatures:  number,
    outFeatures: number,
    options:     LinearOptions = {},
  ) {
    super();
    this.inFeatures  = inFeatures;
    this.outFeatures = outFeatures;
    this._useBias    = options.bias  ?? true;
    this._dtype      = options.dtype ?? "float32";
    this._weightInit = options.weightInit ?? initXavier;
    this._biasInit   = options.biasInit   ?? initZeros;
  }

  protected initParams(): void {
    this._weight = this.addParam("weight", [this.inFeatures, this.outFeatures], this._weightInit, this._dtype);
    if (this._useBias) {
      this._bias = this.addParam("bias", [this.outFeatures], this._biasInit, this._dtype);
    }
  }

  forward(x: SymbolicTensor): SymbolicTensor {
    this._ensureParams();
    const gb = getActiveBuilder();
    const [mm] = gb.applyOp("matmul", [x, this._weight!]);
    if (this._useBias) {
      const [out] = gb.applyOp("add", [mm, this._bias!]);
      return out;
    }
    return mm;
  }

  get weight(): SymbolicTensor {
    if (!this._weight) throw new Error("Linear.weight accessed before forward()");
    return this._weight;
  }

  get bias(): SymbolicTensor | null { return this._bias; }
}
