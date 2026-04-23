// ─────────────────────────────────────────────────────────────────────────────
// frontend/nn/activation.ts
//
// Activation modules: ReLU, Sigmoid, Tanh, GELU, Identity.
// Each is a stateless Module — no parameters, just a single op.
// ─────────────────────────────────────────────────────────────────────────────

import { Module }           from "./module";
import { SymbolicTensor }   from "../tensor/tensor";
import { getActiveBuilder } from "../core/context";

function unaryModule(opName: string): new () => Module {
  return class extends Module {
    forward(x: SymbolicTensor): SymbolicTensor {
      this._ensureParams();
      return getActiveBuilder().applyOp(opName, [x])[0];
    }
  };
}

/** Rectified Linear Unit: max(0, x). */
export const ReLU    = unaryModule("relu");
/** Sigmoid activation: 1 / (1 + exp(-x)). */
export const Sigmoid = unaryModule("sigmoid");
/** Hyperbolic tangent. */
export const Tanh    = unaryModule("tanh");
/** Gaussian Error Linear Unit. */
export const GELU    = unaryModule("gelu");

/** Pass-through identity module. Useful as a placeholder. */
export class Identity extends Module {
  forward(x: SymbolicTensor): SymbolicTensor { return x; }
}
