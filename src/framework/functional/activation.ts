// ─────────────────────────────────────────────────────────────────────────────
// frontend/functional/activation.ts
//
// Activation functional ops. Require an active ExportSession context.
// ─────────────────────────────────────────────────────────────────────────────

import { getActiveBuilder } from "../core/context";
import { SymbolicTensor }   from "../tensor/tensor";

/** Rectified linear unit: max(0, x). */
export function relu(x: SymbolicTensor):    SymbolicTensor { return getActiveBuilder().applyOp("relu",    [x])[0]; }
/** Sigmoid: 1 / (1 + exp(-x)). */
export function sigmoid(x: SymbolicTensor): SymbolicTensor { return getActiveBuilder().applyOp("sigmoid", [x])[0]; }
/** Hyperbolic tangent. */
export function tanh(x: SymbolicTensor):    SymbolicTensor { return getActiveBuilder().applyOp("tanh",    [x])[0]; }
/** Gaussian error linear unit. */
export function gelu(x: SymbolicTensor):    SymbolicTensor { return getActiveBuilder().applyOp("gelu",    [x])[0]; }
