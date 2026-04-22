// ─────────────────────────────────────────────────────────────────────────────
// index.ts
//
// Public API surface + demo runner.
// ─────────────────────────────────────────────────────────────────────────────

// ── IR layer ──────────────────────────────────────────────────────────────────
export { Graph, resetCounters } from "./ir/graph";
export type { Tensor, Node }    from "./ir/graph";
export type { DType, Shape, Attrs, FusionRule, ChainCandidate } from "./ir/types";
export { validateGraph }        from "./ir/validate";
export type { ValidationResult, ValidationError, ValidationErrorKind } from "./ir/validate";

// ── Loop IR ───────────────────────────────────────────────────────────────────
export type {
  LoopModule, LoopFunction, LoopParam, LoopStmt,
  LoopExpr, LoopVar, MemRef, BinOp, CallBuiltin, Literal,
  ForLoop, Assign,
} from "./ir/loopIR";
export {
  loopVar, memRef, binOp, callBuiltin, literal, assign, forLoop, nestedLoops,
} from "./ir/loopIR";

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
export { createDefaultPipeline } from "./passes/pipelines";
export type { DefaultPipeline } from "./passes/pipelines";

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

  console.log("\n  All examples complete.\n");
}
