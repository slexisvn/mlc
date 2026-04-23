// ─────────────────────────────────────────────────────────────────────────────
// frontend/functional/elementwise.ts
//
// Element-wise functional ops. Require an active ExportSession context.
// ─────────────────────────────────────────────────────────────────────────────

import { getActiveBuilder } from "../core/context";
import { SymbolicTensor }   from "../tensor/tensor";

/** Element-wise addition: x + y (broadcasts). */
export function add(x: SymbolicTensor, y: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("add", [x, y])[0];
}
/** Element-wise subtraction: x - y (broadcasts). */
export function sub(x: SymbolicTensor, y: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("sub", [x, y])[0];
}
/** Element-wise multiplication: x * y (broadcasts). */
export function mul(x: SymbolicTensor, y: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("mul", [x, y])[0];
}
/** Element-wise division: x / y (broadcasts). */
export function div(x: SymbolicTensor, y: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("div", [x, y])[0];
}
/** Element-wise negation: -x. */
export function neg(x: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("neg", [x])[0];
}
/** Element-wise absolute value. */
export function abs(x: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("abs", [x])[0];
}
/** Element-wise exponential: e^x. */
export function exp(x: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("exp", [x])[0];
}
/** Element-wise square root. */
export function sqrt(x: SymbolicTensor): SymbolicTensor {
  return getActiveBuilder().applyOp("sqrt", [x])[0];
}
