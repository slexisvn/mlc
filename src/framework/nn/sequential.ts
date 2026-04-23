// ─────────────────────────────────────────────────────────────────────────────
//
// Sequential — applies a list of modules in order.
// ─────────────────────────────────────────────────────────────────────────────

import { Module }         from "./module";
import { SymbolicTensor } from "../tensor/tensor";

/**
 * Container that applies modules sequentially: output of each module is the
 * input to the next.
 *
 * Example:
 * ```ts
 * const mlp = new nn.Sequential(
 *   new nn.Linear(784, 256), new nn.ReLU(),
 *   new nn.Linear(256, 128), new nn.ReLU(),
 *   new nn.Linear(128,  10),
 * );
 * ```
 */
export class Sequential extends Module {
  private readonly _layers: Module[];

  constructor(...layers: Module[]) {
    super();
    this._layers = layers;
    for (let i = 0; i < layers.length; i++) {
      this.register(String(i), layers[i]);
    }
  }

  forward(x: SymbolicTensor): SymbolicTensor {
    this._ensureParams();
    let h: SymbolicTensor = x;
    for (const layer of this._layers) {
      const out = layer.forward(h);
      h = Array.isArray(out) ? out[0] : out;
    }
    return h;
  }

  get layers(): readonly Module[] { return this._layers; }
}
