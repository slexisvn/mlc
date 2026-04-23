// frontend/core/index.ts — internal core barrel (advanced/compiler-facing use)
export { GraphBuilder }                              from "./graphBuilder";
export type { GraphBuilderOptions }                  from "./graphBuilder";
export { OpSchemaRegistry, defaultOpRegistry, DEFAULT_OP_SCHEMAS } from "./opRegistry";
export type { OpSchema, InputSpec, OutputSpec, InferContext, GradBuilderFn } from "./opRegistry";
export { broadcast, matmulShape, transposeShape, reshapeShape, reduceShape, shapeNumel, shapesEqual } from "./shape";
export type { ShapeExpr, Dim }                       from "./shape";
export { FrameworkError, ShapeError, OpError, GraphBuildError, AutodiffError, ContextError } from "./errors";
export type { FrameworkErrorKind }                   from "./errors";
export { getActiveBuilder, getActiveParamSink, hasActiveContext, withActiveContext } from "./context";
export type { ParamSpec }                            from "./context";
