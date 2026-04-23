// frontend/nn/index.ts — public nn namespace barrel
export { Module }               from "./module";
export { Linear }               from "./linear";
export type { LinearOptions }   from "./linear";
export { Sequential }           from "./sequential";
export { ReLU, Sigmoid, Tanh, GELU, Softmax, Identity } from "./activation";
export { initXavier, initZeros, initOnes, initConstant } from "./parameter";
export type { Initialiser, ParameterSpec }     from "./parameter";
