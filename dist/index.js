"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// index.ts
//
// Public API surface + demo runner.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.LayoutRuleRegistry = exports.matchChains = exports.DEFAULT_FUSION_RULES = exports.RuleRegistry = exports.createFusedNode = exports.extractChainBoundary = exports.buildConsumerMap = exports.topoSort = exports.DEFAULT_CONTRACT_REGISTRY = exports.DEFAULT_OP_CONTRACTS = exports.OpContractRegistry = exports.formatTransform = exports.getTransformFromAttrs = exports.PERM_NHWC_TO_NCHW = exports.PERM_NCHW_TO_NHWC = exports.composeTransforms = exports.areInverseTransforms = exports.makeIdentityTransform = exports.makePermutationTransform = exports.isIdentityPermutation = exports.composePermutations = exports.invertPermutation = exports.isValidPermutation = exports.Layouts = exports.analyzeTilingCandidates = exports.analyzeFusionCandidates = exports.stripMine = exports.rebuildNest = exports.substituteStmt = exports.substituteExpr = exports.isReductionLoop = exports.collectWrites = exports.collectReads = exports.collectExprBuffers = exports.isSameIterSpace = exports.isPerfectNest = exports.extractPerfectNest = exports.validateLoopModule = exports.nestedLoops = exports.forLoopDyn = exports.forLoop = exports.assign = exports.literal = exports.callBuiltin = exports.binOp = exports.memRef = exports.loopVar = exports.validateGraph = exports.resetCounters = exports.Graph = void 0;
exports.printLoopModule = exports.printFusionAnalysis = exports.printLayoutAnalysis = exports.printDiff = exports.printExecutionPlan = exports.printGraph = exports.DeadCodeEliminationPass = exports.CSEPass = exports.resetCFCounter = exports.ConstantFoldingPass = exports.DEFAULT_TILING_CONFIG = exports.LoopTilingPass = exports.DEFAULT_LOOP_FUSION_CONFIG = exports.LoopFusionPass = exports.LoopPassManager = exports.createFullPipeline = exports.createLoopPipeline = exports.createDefaultPipeline = exports.LoopLoweringPass = exports.LayoutTransformPass = exports.FusionPass = exports.PassManager = exports.opAcceptsLayout = exports.canPropagateLayout = exports.simplifyPair = exports.transformsCancel = exports.simplifyTransformChain = exports.DEFAULT_COST_MODEL_CONFIG = exports.CostModel = exports.analyzeLayouts = exports.analyzeFusion = exports.matchLayoutChains = exports.DEFAULT_LAYOUT_RULES = void 0;
// ── IR layer ──────────────────────────────────────────────────────────────────
var graph_1 = require("./ir/graph");
Object.defineProperty(exports, "Graph", { enumerable: true, get: function () { return graph_1.Graph; } });
Object.defineProperty(exports, "resetCounters", { enumerable: true, get: function () { return graph_1.resetCounters; } });
var validate_1 = require("./ir/validate");
Object.defineProperty(exports, "validateGraph", { enumerable: true, get: function () { return validate_1.validateGraph; } });
var loopIR_1 = require("./ir/loopIR");
Object.defineProperty(exports, "loopVar", { enumerable: true, get: function () { return loopIR_1.loopVar; } });
Object.defineProperty(exports, "memRef", { enumerable: true, get: function () { return loopIR_1.memRef; } });
Object.defineProperty(exports, "binOp", { enumerable: true, get: function () { return loopIR_1.binOp; } });
Object.defineProperty(exports, "callBuiltin", { enumerable: true, get: function () { return loopIR_1.callBuiltin; } });
Object.defineProperty(exports, "literal", { enumerable: true, get: function () { return loopIR_1.literal; } });
Object.defineProperty(exports, "assign", { enumerable: true, get: function () { return loopIR_1.assign; } });
Object.defineProperty(exports, "forLoop", { enumerable: true, get: function () { return loopIR_1.forLoop; } });
Object.defineProperty(exports, "forLoopDyn", { enumerable: true, get: function () { return loopIR_1.forLoopDyn; } });
Object.defineProperty(exports, "nestedLoops", { enumerable: true, get: function () { return loopIR_1.nestedLoops; } });
// ── Loop IR validator ─────────────────────────────────────────────────────────
var validateLoop_1 = require("./ir/validateLoop");
Object.defineProperty(exports, "validateLoopModule", { enumerable: true, get: function () { return validateLoop_1.validateLoopModule; } });
// ── Loop IR analysis ─────────────────────────────────────────────────────────
var loopAnalysis_1 = require("./analysis/loopAnalysis");
Object.defineProperty(exports, "extractPerfectNest", { enumerable: true, get: function () { return loopAnalysis_1.extractPerfectNest; } });
Object.defineProperty(exports, "isPerfectNest", { enumerable: true, get: function () { return loopAnalysis_1.isPerfectNest; } });
Object.defineProperty(exports, "isSameIterSpace", { enumerable: true, get: function () { return loopAnalysis_1.isSameIterSpace; } });
Object.defineProperty(exports, "collectExprBuffers", { enumerable: true, get: function () { return loopAnalysis_1.collectExprBuffers; } });
Object.defineProperty(exports, "collectReads", { enumerable: true, get: function () { return loopAnalysis_1.collectReads; } });
Object.defineProperty(exports, "collectWrites", { enumerable: true, get: function () { return loopAnalysis_1.collectWrites; } });
Object.defineProperty(exports, "isReductionLoop", { enumerable: true, get: function () { return loopAnalysis_1.isReductionLoop; } });
Object.defineProperty(exports, "substituteExpr", { enumerable: true, get: function () { return loopAnalysis_1.substituteExpr; } });
Object.defineProperty(exports, "substituteStmt", { enumerable: true, get: function () { return loopAnalysis_1.substituteStmt; } });
Object.defineProperty(exports, "rebuildNest", { enumerable: true, get: function () { return loopAnalysis_1.rebuildNest; } });
Object.defineProperty(exports, "stripMine", { enumerable: true, get: function () { return loopAnalysis_1.stripMine; } });
Object.defineProperty(exports, "analyzeFusionCandidates", { enumerable: true, get: function () { return loopAnalysis_1.analyzeFusionCandidates; } });
Object.defineProperty(exports, "analyzeTilingCandidates", { enumerable: true, get: function () { return loopAnalysis_1.analyzeTilingCandidates; } });
// ── Layout model ──────────────────────────────────────────────────────────────
var layouts_1 = require("./ir/layouts");
Object.defineProperty(exports, "Layouts", { enumerable: true, get: function () { return layouts_1.Layouts; } });
Object.defineProperty(exports, "isValidPermutation", { enumerable: true, get: function () { return layouts_1.isValidPermutation; } });
Object.defineProperty(exports, "invertPermutation", { enumerable: true, get: function () { return layouts_1.invertPermutation; } });
Object.defineProperty(exports, "composePermutations", { enumerable: true, get: function () { return layouts_1.composePermutations; } });
Object.defineProperty(exports, "isIdentityPermutation", { enumerable: true, get: function () { return layouts_1.isIdentityPermutation; } });
Object.defineProperty(exports, "makePermutationTransform", { enumerable: true, get: function () { return layouts_1.makePermutationTransform; } });
Object.defineProperty(exports, "makeIdentityTransform", { enumerable: true, get: function () { return layouts_1.makeIdentityTransform; } });
Object.defineProperty(exports, "areInverseTransforms", { enumerable: true, get: function () { return layouts_1.areInverseTransforms; } });
Object.defineProperty(exports, "composeTransforms", { enumerable: true, get: function () { return layouts_1.composeTransforms; } });
Object.defineProperty(exports, "PERM_NCHW_TO_NHWC", { enumerable: true, get: function () { return layouts_1.PERM_NCHW_TO_NHWC; } });
Object.defineProperty(exports, "PERM_NHWC_TO_NCHW", { enumerable: true, get: function () { return layouts_1.PERM_NHWC_TO_NCHW; } });
Object.defineProperty(exports, "getTransformFromAttrs", { enumerable: true, get: function () { return layouts_1.getTransformFromAttrs; } });
Object.defineProperty(exports, "formatTransform", { enumerable: true, get: function () { return layouts_1.formatTransform; } });
// ── Op contracts ──────────────────────────────────────────────────────────────
var opContracts_1 = require("./ops/opContracts");
Object.defineProperty(exports, "OpContractRegistry", { enumerable: true, get: function () { return opContracts_1.OpContractRegistry; } });
Object.defineProperty(exports, "DEFAULT_OP_CONTRACTS", { enumerable: true, get: function () { return opContracts_1.DEFAULT_OP_CONTRACTS; } });
Object.defineProperty(exports, "DEFAULT_CONTRACT_REGISTRY", { enumerable: true, get: function () { return opContracts_1.DEFAULT_CONTRACT_REGISTRY; } });
// ── Utilities ────────────────────────────────────────────────────────────────
var toposort_1 = require("./utils/toposort");
Object.defineProperty(exports, "topoSort", { enumerable: true, get: function () { return toposort_1.topoSort; } });
var graphUtils_1 = require("./utils/graphUtils");
Object.defineProperty(exports, "buildConsumerMap", { enumerable: true, get: function () { return graphUtils_1.buildConsumerMap; } });
Object.defineProperty(exports, "extractChainBoundary", { enumerable: true, get: function () { return graphUtils_1.extractChainBoundary; } });
Object.defineProperty(exports, "createFusedNode", { enumerable: true, get: function () { return graphUtils_1.createFusedNode; } });
// ── Pattern system ────────────────────────────────────────────────────────────
var rules_1 = require("./patterns/rules");
Object.defineProperty(exports, "RuleRegistry", { enumerable: true, get: function () { return rules_1.RuleRegistry; } });
Object.defineProperty(exports, "DEFAULT_FUSION_RULES", { enumerable: true, get: function () { return rules_1.DEFAULT_FUSION_RULES; } });
var matcher_1 = require("./patterns/matcher");
Object.defineProperty(exports, "matchChains", { enumerable: true, get: function () { return matcher_1.matchChains; } });
var layoutRules_1 = require("./patterns/layoutRules");
Object.defineProperty(exports, "LayoutRuleRegistry", { enumerable: true, get: function () { return layoutRules_1.LayoutRuleRegistry; } });
Object.defineProperty(exports, "DEFAULT_LAYOUT_RULES", { enumerable: true, get: function () { return layoutRules_1.DEFAULT_LAYOUT_RULES; } });
var layoutMatcher_1 = require("./patterns/layoutMatcher");
Object.defineProperty(exports, "matchLayoutChains", { enumerable: true, get: function () { return layoutMatcher_1.matchLayoutChains; } });
// ── Analysis ─────────────────────────────────────────────────────────────────
var fusionAnalysis_1 = require("./analysis/fusionAnalysis");
Object.defineProperty(exports, "analyzeFusion", { enumerable: true, get: function () { return fusionAnalysis_1.analyzeFusion; } });
var layoutAnalysis_1 = require("./analysis/layoutAnalysis");
Object.defineProperty(exports, "analyzeLayouts", { enumerable: true, get: function () { return layoutAnalysis_1.analyzeLayouts; } });
// ── Optimizer ─────────────────────────────────────────────────────────────────
var costModel_1 = require("./optimizer/costModel");
Object.defineProperty(exports, "CostModel", { enumerable: true, get: function () { return costModel_1.CostModel; } });
Object.defineProperty(exports, "DEFAULT_COST_MODEL_CONFIG", { enumerable: true, get: function () { return costModel_1.DEFAULT_COST_MODEL_CONFIG; } });
var layoutCanonicalizer_1 = require("./optimizer/layoutCanonicalizer");
Object.defineProperty(exports, "simplifyTransformChain", { enumerable: true, get: function () { return layoutCanonicalizer_1.simplifyTransformChain; } });
Object.defineProperty(exports, "transformsCancel", { enumerable: true, get: function () { return layoutCanonicalizer_1.transformsCancel; } });
Object.defineProperty(exports, "simplifyPair", { enumerable: true, get: function () { return layoutCanonicalizer_1.simplifyPair; } });
var layoutPropagation_1 = require("./optimizer/layoutPropagation");
Object.defineProperty(exports, "canPropagateLayout", { enumerable: true, get: function () { return layoutPropagation_1.canPropagateLayout; } });
Object.defineProperty(exports, "opAcceptsLayout", { enumerable: true, get: function () { return layoutPropagation_1.opAcceptsLayout; } });
var passManager_1 = require("./passes/passManager");
Object.defineProperty(exports, "PassManager", { enumerable: true, get: function () { return passManager_1.PassManager; } });
var fusionPass_1 = require("./passes/fusionPass");
Object.defineProperty(exports, "FusionPass", { enumerable: true, get: function () { return fusionPass_1.FusionPass; } });
var layoutTransformPass_1 = require("./passes/layoutTransformPass");
Object.defineProperty(exports, "LayoutTransformPass", { enumerable: true, get: function () { return layoutTransformPass_1.LayoutTransformPass; } });
var loopLoweringPass_1 = require("./passes/loopLoweringPass");
Object.defineProperty(exports, "LoopLoweringPass", { enumerable: true, get: function () { return loopLoweringPass_1.LoopLoweringPass; } });
var pipelines_1 = require("./passes/pipelines");
Object.defineProperty(exports, "createDefaultPipeline", { enumerable: true, get: function () { return pipelines_1.createDefaultPipeline; } });
Object.defineProperty(exports, "createLoopPipeline", { enumerable: true, get: function () { return pipelines_1.createLoopPipeline; } });
Object.defineProperty(exports, "createFullPipeline", { enumerable: true, get: function () { return pipelines_1.createFullPipeline; } });
var loopPass_1 = require("./passes/loopPass");
Object.defineProperty(exports, "LoopPassManager", { enumerable: true, get: function () { return loopPass_1.LoopPassManager; } });
var loopFusionPass_1 = require("./passes/loopFusionPass");
Object.defineProperty(exports, "LoopFusionPass", { enumerable: true, get: function () { return loopFusionPass_1.LoopFusionPass; } });
Object.defineProperty(exports, "DEFAULT_LOOP_FUSION_CONFIG", { enumerable: true, get: function () { return loopFusionPass_1.DEFAULT_LOOP_FUSION_CONFIG; } });
var loopTilingPass_1 = require("./passes/loopTilingPass");
Object.defineProperty(exports, "LoopTilingPass", { enumerable: true, get: function () { return loopTilingPass_1.LoopTilingPass; } });
Object.defineProperty(exports, "DEFAULT_TILING_CONFIG", { enumerable: true, get: function () { return loopTilingPass_1.DEFAULT_TILING_CONFIG; } });
var constantFoldingPass_1 = require("./passes/constantFoldingPass");
Object.defineProperty(exports, "ConstantFoldingPass", { enumerable: true, get: function () { return constantFoldingPass_1.ConstantFoldingPass; } });
Object.defineProperty(exports, "resetCFCounter", { enumerable: true, get: function () { return constantFoldingPass_1.resetCFCounter; } });
var csePass_1 = require("./passes/csePass");
Object.defineProperty(exports, "CSEPass", { enumerable: true, get: function () { return csePass_1.CSEPass; } });
var deadCodeEliminationPass_1 = require("./passes/deadCodeEliminationPass");
Object.defineProperty(exports, "DeadCodeEliminationPass", { enumerable: true, get: function () { return deadCodeEliminationPass_1.DeadCodeEliminationPass; } });
// ── Debug utilities ───────────────────────────────────────────────────────────
var printer_1 = require("./debug/printer");
Object.defineProperty(exports, "printGraph", { enumerable: true, get: function () { return printer_1.printGraph; } });
Object.defineProperty(exports, "printExecutionPlan", { enumerable: true, get: function () { return printer_1.printExecutionPlan; } });
Object.defineProperty(exports, "printDiff", { enumerable: true, get: function () { return printer_1.printDiff; } });
Object.defineProperty(exports, "printLayoutAnalysis", { enumerable: true, get: function () { return printer_1.printLayoutAnalysis; } });
Object.defineProperty(exports, "printFusionAnalysis", { enumerable: true, get: function () { return printer_1.printFusionAnalysis; } });
var loopPrinter_1 = require("./debug/loopPrinter");
Object.defineProperty(exports, "printLoopModule", { enumerable: true, get: function () { return loopPrinter_1.printLoopModule; } });
// ─────────────────────────────────────────────────────────────────────────────
// Demo runner — executed only when this file is the program entry point.
// ─────────────────────────────────────────────────────────────────────────────
const simpleChain_1 = require("./examples/simpleChain");
const branching_1 = require("./examples/branching");
const nOpChain_1 = require("./examples/nOpChain");
const layoutCancellation_1 = require("./examples/layoutCancellation");
const layoutPropagationExample_1 = require("./examples/layoutPropagationExample");
const layoutMismatch_1 = require("./examples/layoutMismatch");
const mixedPipeline_1 = require("./examples/mixedPipeline");
const loopFusion_1 = require("./examples/loopFusion");
const loopTiling_1 = require("./examples/loopTiling");
const loopOptimization_1 = require("./examples/loopOptimization");
const preLayoutOptimization_1 = require("./examples/preLayoutOptimization");
if (require.main === module) {
    console.log("\n╔══════════════════════════════════════════════════════════╗");
    console.log("║     MINI ML COMPILER  ·  Fusion + Layout Demo            ║");
    console.log("╚══════════════════════════════════════════════════════════╝");
    console.log("\n── Original fusion examples ────────────────────────────────");
    (0, simpleChain_1.runSimpleChainExample)();
    (0, branching_1.runBranchingExample)();
    (0, nOpChain_1.runNOpChainExample)();
    console.log("\n── Layout transform examples ───────────────────────────────");
    (0, layoutCancellation_1.runLayoutCancellationExample)();
    (0, layoutPropagationExample_1.runLayoutPropagationExample)();
    (0, layoutMismatch_1.runLayoutMismatchExample)();
    (0, mixedPipeline_1.runMixedPipelineExample)();
    console.log("\n── Loop IR optimization examples ───────────────────────────");
    (0, loopFusion_1.runLoopFusionExample)();
    (0, loopTiling_1.runLoopTilingExample)();
    (0, loopOptimization_1.runLoopOptimizationExample)();
    console.log("\n── Pre-layout simplification examples ──────────────────────");
    (0, preLayoutOptimization_1.runPreLayoutOptimizationExample)();
    console.log("\n  All examples complete.\n");
}
//# sourceMappingURL=index.js.map