// ─────────────────────────────────────────────────────────────────────────────
// index.ts
//
// Public API surface + demo runner.
// ─────────────────────────────────────────────────────────────────────────────
import { stdout } from "process";
stdout.setDefaultEncoding("utf8");
// ── IR layer ──────────────────────────────────────────────────────────────────
export { Graph, resetCounters } from "./ir/graph";
export type { Tensor, Node }    from "./ir/graph";
export type { DType, Shape, Attrs, FusionRule, ChainCandidate, ConstantPayload } from "./ir/types";
export { validateGraph }        from "./ir/validate";
export type { ValidationResult, ValidationError, ValidationErrorKind } from "./ir/validate";

// ── Loop IR ───────────────────────────────────────────────────────────────────
export type {
  LoopModule, LoopFunction, LoopParam, LoopStmt,
  LoopExpr, LoopVar, MemRef, BinOp, CallBuiltin, Literal,
  ForLoop, Assign,
} from "./ir/loopIR";
export {
  loopVar, memRef, binOp, callBuiltin, literal, assign, forLoop, forLoopDyn, nestedLoops,
} from "./ir/loopIR";

// ── Loop IR validator ─────────────────────────────────────────────────────────
export { validateLoopModule } from "./ir/validateLoop";
export type {
  LoopValidationResult, LoopValidationError, LoopValidationErrorKind,
} from "./ir/validateLoop";

// ── Loop IR analysis ─────────────────────────────────────────────────────────
export {
  extractPerfectNest, isPerfectNest, isSameIterSpace,
  collectExprBuffers, collectReads, collectWrites,
  isReductionLoop,
  substituteExpr, substituteStmt,
  rebuildNest, stripMine,
  analyzeFusionCandidates, analyzeTilingCandidates,
} from "./analysis/loopAnalysis";
export type {
  PerfectNestLevel, PerfectNestInfo,
  FusionCandidate, FusionRejection, LoopFusionAnalysis,
  TilingCandidate, TilingRejection, LoopTilingAnalysis,
} from "./analysis/loopAnalysis";

// ── Layout model ──────────────────────────────────────────────────────────────
export {
  Layouts,
  isValidPermutation, invertPermutation, composePermutations, isIdentityPermutation,
  makePermutationTransform, makeIdentityTransform,
  areInverseTransforms, composeTransforms,
  PERM_NCHW_TO_NHWC, PERM_NHWC_TO_NCHW,
  getTransformFromAttrs, formatTransform,
} from "./ir/layouts";
export type { LayoutFormat, Permutation, LayoutTransform, LayoutTransformKind } from "./ir/layouts";

// ── Op contracts ──────────────────────────────────────────────────────────────
export { OpContractRegistry, DEFAULT_OP_CONTRACTS, DEFAULT_CONTRACT_REGISTRY } from "./ops/opContracts";
export type { OpContract, OpLayoutBehavior, FusibilityClass } from "./ops/opContracts";

// ── Utilities ────────────────────────────────────────────────────────────────
export { topoSort }             from "./utils/toposort";
export type { TopoSortResult }  from "./utils/toposort";
export { buildConsumerMap, extractChainBoundary, createFusedNode } from "./utils/graphUtils";

// ── Pattern system ────────────────────────────────────────────────────────────
export { RuleRegistry, DEFAULT_FUSION_RULES } from "./patterns/rules";
export { matchChains }          from "./patterns/matcher";
export type { MatchedChain }    from "./patterns/matcher";
export { LayoutRuleRegistry, DEFAULT_LAYOUT_RULES } from "./patterns/layoutRules";
export type { LayoutRewriteRule, LayoutRuleKind }   from "./patterns/layoutRules";
export { matchLayoutChains }    from "./patterns/layoutMatcher";
export type { MatchedLayoutChain } from "./patterns/layoutMatcher";

// ── Analysis ─────────────────────────────────────────────────────────────────
export { analyzeFusion }        from "./analysis/fusionAnalysis";
export type {
  FusionAnalysisResult, CandidateChain, RejectionRecord, RejectionReason,
} from "./analysis/fusionAnalysis";
export { analyzeLayouts }       from "./analysis/layoutAnalysis";
export type {
  LayoutAnalysisResult, TensorLayoutFact, LayoutConflict, EliminationCandidate,
} from "./analysis/layoutAnalysis";

// ── Optimizer ─────────────────────────────────────────────────────────────────
export { CostModel, DEFAULT_COST_MODEL_CONFIG } from "./optimizer/costModel";
export type { CostModelConfig, CostModelEvaluation } from "./optimizer/costModel";
export { simplifyTransformChain, transformsCancel, simplifyPair } from "./optimizer/layoutCanonicalizer";
export type { CanonicalizationResult } from "./optimizer/layoutCanonicalizer";
export { canPropagateLayout, opAcceptsLayout } from "./optimizer/layoutPropagation";
export type { PropagationResult } from "./optimizer/layoutPropagation";

// ── Pass system ───────────────────────────────────────────────────────────────
export type { Pass, PassLog, PassResult } from "./passes/pass";
export { PassManager }          from "./passes/passManager";
export type { PassManagerOptions } from "./passes/passManager";
export { FusionPass }           from "./passes/fusionPass";
export { LayoutTransformPass }  from "./passes/layoutTransformPass";
export { LoopLoweringPass }     from "./passes/loopLoweringPass";
export { createDefaultPipeline, createLoopPipeline, createFullPipeline } from "./passes/pipelines";
export type { DefaultPipeline, LoopPipelineOptions, FullPipeline } from "./passes/pipelines";
export { LoopPassManager }    from "./passes/loopPass";
export type { LoopPass, LoopPassResult, LoopPassManagerOptions } from "./passes/loopPass";
export { LoopFusionPass, DEFAULT_LOOP_FUSION_CONFIG } from "./passes/loopFusionPass";
export type { LoopFusionConfig } from "./passes/loopFusionPass";
export { LoopTilingPass, DEFAULT_TILING_CONFIG } from "./passes/loopTilingPass";
export type { TilingConfig } from "./passes/loopTilingPass";
export { ConstantFoldingPass, resetCFCounter } from "./passes/constantFoldingPass";
export { CSEPass }                             from "./passes/csePass";
export { DeadCodeEliminationPass }             from "./passes/deadCodeEliminationPass";

// ── Debug utilities ───────────────────────────────────────────────────────────
export {
  printGraph, printExecutionPlan, printDiff,
  printLayoutAnalysis, printFusionAnalysis,
} from "./debug/printer";
export { printLoopModule } from "./debug/loopPrinter";

// ─────────────────────────────────────────────────────────────────────────────
// Demo runner — executed only when this file is the program entry point.
// ─────────────────────────────────────────────────────────────────────────────

import { runSimpleChainExample }        from "./examples/simpleChain";
import { runBranchingExample }           from "./examples/branching";
import { runNOpChainExample }            from "./examples/nOpChain";
import { runLayoutCancellationExample }  from "./examples/layoutCancellation";
import { runLayoutPropagationExample }   from "./examples/layoutPropagationExample";
import { runLayoutMismatchExample }      from "./examples/layoutMismatch";
import { runMixedPipelineExample }       from "./examples/mixedPipeline";
import { runLoopFusionExample }          from "./examples/loopFusion";
import { runLoopTilingExample }          from "./examples/loopTiling";
import { runLoopOptimizationExample }    from "./examples/loopOptimization";
import { runPreLayoutOptimizationExample } from "./examples/preLayoutOptimization";

if (require.main === module) {
  console.log("\n╔══════════════════════════════════════════════════════════╗");
  console.log("║     MINI ML COMPILER  ·  Fusion + Layout Demo            ║");
  console.log("╚══════════════════════════════════════════════════════════╝");

  console.log("\n── Original fusion examples ────────────────────────────────");
  runSimpleChainExample();
  runBranchingExample();
  runNOpChainExample();

  console.log("\n── Layout transform examples ───────────────────────────────");
  runLayoutCancellationExample();
  runLayoutPropagationExample();
  runLayoutMismatchExample();
  runMixedPipelineExample();

  console.log("\n── Loop IR optimization examples ───────────────────────────");
  runLoopFusionExample();
  runLoopTilingExample();
  runLoopOptimizationExample();

  console.log("\n── Pre-layout simplification examples ──────────────────────");
  runPreLayoutOptimizationExample();

  console.log("\n  All examples complete.\n");
}
