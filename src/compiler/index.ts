// ─────────────────────────────────────────────────────────────────────────────
// compiler/index.ts — public API for the MLC compiler
// ─────────────────────────────────────────────────────────────────────────────

// IR
export { Graph, resetCounters }                        from "./ir/graph";
export type { Tensor as CompilerTensor, Node as CompilerNode } from "./ir/graph";
export { validateGraph }                               from "./ir/validate";
export type { ValidationResult, ValidationError }      from "./ir/validate";

// Pass infrastructure
export { PassManager }                                 from "./passes/passManager";
export type { PassManagerOptions }                     from "./passes/passManager";
export type { Pass, PassResult, PassLog, LogLevel }    from "./passes/pass";

// Pipelines
export {
  createDefaultPipeline,
  createLoopPipeline,
  createFullPipeline,
}                                                      from "./passes/pipelines";
export type {
  DefaultPipeline,
  LoopPipelineOptions,
  FullPipeline,
  LoopModule,
}                                                      from "./passes/pipelines";

// Individual graph passes
export { ConstantFoldingPass }                         from "./passes/constantFoldingPass";
export { CSEPass }                                     from "./passes/csePass";
export { DeadCodeEliminationPass }                     from "./passes/deadCodeEliminationPass";
export { LayoutTransformPass }                         from "./passes/layoutTransformPass";
export { FusionPass }                                  from "./passes/fusionPass";
export { LoopLoweringPass }                            from "./passes/loopLoweringPass";

// Loop passes
export { LoopPassManager }                             from "./passes/loopPass";
export { LoopFusionPass }                              from "./passes/loopFusionPass";
export { LoopTilingPass }                              from "./passes/loopTilingPass";

// Debug / pretty-print utilities
export {
  printGraph,
  printExecutionPlan,
  printDiff,
  printLayoutAnalysis,
  printFusionAnalysis,
}                                                      from "./debug/printer";
export { printLoopModule }                             from "./debug/loopPrinter";
