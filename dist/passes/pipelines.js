"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// passes/pipelines.ts
//
// Pre-built pipeline factories with the correct default pass ordering.
//
// Default order (must be respected for correctness):
//   1. ConstantFoldingPass     — evaluate foldable ops whose inputs are
//      compile-time constants; replace compute nodes with "const" source nodes.
//   2. CSEPass                 — deduplicate nodes that produce identical values
//      (same op + attrs + inputs); rewire consumers to the canonical producer.
//   3. DeadCodeEliminationPass — remove nodes and tensors not reachable from
//      any graph output (includes formerly-live nodes orphaned by CF / CSE).
//   4. LayoutTransformPass     — cancel redundant transpose pairs and propagate
//      layout information so downstream passes see a clean graph.
//   5. FusionPass              — fuse adjacent elementwise and matmul chains
//      after layout noise has been removed.
//   6. LoopLoweringPass        — terminal pass; translates the optimized Graph
//      IR to an explicit Loop IR without mutating the graph.
//
// Usage:
//   const { pm, loopPass } = createDefaultPipeline();
//   const optimized = pm.run(myGraph);
//   const loopModule = loopPass.getLastModule();
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.createDefaultPipeline = createDefaultPipeline;
exports.createLoopPipeline = createLoopPipeline;
exports.createFullPipeline = createFullPipeline;
const passManager_1 = require("./passManager");
const constantFoldingPass_1 = require("./constantFoldingPass");
const csePass_1 = require("./csePass");
const deadCodeEliminationPass_1 = require("./deadCodeEliminationPass");
const layoutTransformPass_1 = require("./layoutTransformPass");
const fusionPass_1 = require("./fusionPass");
const loopLoweringPass_1 = require("./loopLoweringPass");
const layoutRules_1 = require("../patterns/layoutRules");
const rules_1 = require("../patterns/rules");
const costModel_1 = require("../optimizer/costModel");
const opContracts_1 = require("../ops/opContracts");
const loopPass_1 = require("./loopPass");
const loopFusionPass_1 = require("./loopFusionPass");
const loopTilingPass_1 = require("./loopTilingPass");
/**
 * Create a PassManager preconfigured with the six default passes in the
 * correct order:
 *   ConstantFoldingPass → CSEPass → DeadCodeEliminationPass
 *   → LayoutTransformPass → FusionPass → LoopLoweringPass
 *
 * All passes are instantiated with their default registries and cost models.
 * To customize individual passes, build the pipeline manually with
 * `PassManager.addPasses()`.
 *
 * @param options  Forwarded to PassManager.  `validateAfterEachPass` defaults
 *                 to `true` as recommended for development.
 */
function createDefaultPipeline(options = {}) {
    const loopPass = new loopLoweringPass_1.LoopLoweringPass();
    const pm = new passManager_1.PassManager({ validateAfterEachPass: true, ...options });
    pm.addPasses(
    // Pre-layout simplification: constant folding → CSE → DCE.
    // Running these three passes before layout analysis ensures that
    // LayoutTransformPass and FusionPass operate on the smallest, cleanest
    // possible graph: no redundant constant computations, no duplicate
    // subexpressions, no dead nodes.
    new constantFoldingPass_1.ConstantFoldingPass(opContracts_1.DEFAULT_CONTRACT_REGISTRY), new csePass_1.CSEPass(opContracts_1.DEFAULT_CONTRACT_REGISTRY), new deadCodeEliminationPass_1.DeadCodeEliminationPass(), 
    // Layout-aware rewrites and operator fusion.
    new layoutTransformPass_1.LayoutTransformPass(new layoutRules_1.LayoutRuleRegistry(), opContracts_1.DEFAULT_CONTRACT_REGISTRY), new fusionPass_1.FusionPass(new rules_1.RuleRegistry(rules_1.DEFAULT_FUSION_RULES), new costModel_1.CostModel()), 
    // Loop IR lowering (terminal — does not mutate the graph).
    loopPass);
    return { pm, loopPass };
}
/**
 * Create a LoopPassManager preconfigured with LoopFusionPass followed by
 * LoopTilingPass — the recommended order (fuse adjacent nests first, then
 * tile the resulting larger nests for cache locality).
 *
 * Usage:
 *   const loopPm = createLoopPipeline();
 *   const optimizedModule = loopPm.run(loopPass.getLastModule()!);
 */
function createLoopPipeline(options = {}) {
    const pm = new loopPass_1.LoopPassManager({ validateAfterEachPass: true, ...options.manager });
    pm.addPasses(new loopFusionPass_1.LoopFusionPass(options.fusion), new loopTilingPass_1.LoopTilingPass(options.tiling));
    return pm;
}
/**
 * Create a full two-stage pipeline: graph optimisation followed by loop
 * IR optimisation.
 *
 * The graph pipeline is identical to `createDefaultPipeline()`.
 * The loop pipeline is identical to `createLoopPipeline()`.
 *
 * @param graphOptions  Forwarded to `createDefaultPipeline()`.
 * @param loopOptions   Forwarded to `createLoopPipeline()`.
 */
function createFullPipeline(graphOptions = {}, loopOptions = {}) {
    const { pm, loopPass } = createDefaultPipeline(graphOptions);
    const loopPm = createLoopPipeline(loopOptions);
    return { pm, loopPass, loopPm };
}
//# sourceMappingURL=pipelines.js.map