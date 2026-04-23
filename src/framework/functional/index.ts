// frontend/functional/index.ts — public functional namespace barrel
export { add, sub, mul, div, neg, abs, exp, sqrt } from "./elementwise";
export { relu, sigmoid, tanh, gelu }               from "./activation";
export { matmul, linear }                          from "./linear";
export { sum, mean }                               from "./reduction";
export type { ReduceOptions }                      from "./reduction";
export { reshape, transpose }                      from "./shape";
