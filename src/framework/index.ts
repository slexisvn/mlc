// frontend/index.ts — top-level frontend barrel
// Import with:  import * as nn from "./frontend/nn"  etc.
// Or via root:  import { nn, functional, ExportSession } from "./frontend"

export * as nn         from "./nn";
export * as functional from "./functional";
export { ExportSession, SessionContext } from "./export/session";
export type { ExportSessionOptions }     from "./export/session";
export { SymbolicTensor }                from "./tensor/tensor";
export type { Tensor }                   from "./tensor/tensor";
export { buildBackwardGraph, DEFAULT_GRAD_BUILDERS, GradRegistry, defaultGradRegistry } from "./autodiff";
export type { BackwardResult, GradBuilderFn, GradContext } from "./autodiff";
// Bridge / IR utilities — re-exported for single-import convenience
export { importGraphIR, importSingleGraphIR, BridgeError } from "./export/importGraphIR";
export type { ImportResult, ImportOptions }                from "./export/importGraphIR";
export { serializeToJSON, deserializeFromJSON, DeserializationError } from "./ir/serializer";
export type { SerializeOptions }                          from "./ir/serializer";
export { validateIRPackage }                              from "./ir/validator";
export type { IRValidationResult, IRValidationError, IRValidationErrorKind } from "./ir/validator";
