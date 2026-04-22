"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/mixedPipeline.ts
//
// End-to-end demo: LayoutTransformPass → FusionPass → LoopLoweringPass.
//
// Graph (before):
//   x[NCHW], bias → transpose(NCHW→NHWC)
//                 → add(x_nhwc, bias)
//                 → transpose(NHWC→NCHW)
//                 → relu
//                 → [output]
//
// After LayoutTransformPass:
//   x[NCHW], bias → add → relu → [output]
//   (sandwich transposes cancelled by propagation rule)
//
// After FusionPass:
//   x[NCHW], bias → add_relu → [output]
//   (add→relu fused by the add_relu rule)
//
// After LoopLoweringPass (terminal — graph unchanged):
//   One LoopFunction with a single fused loop nest — no intermediate buffers.
//   out[i0,i1,i2,i3] = max(0.0, (x_NCHW[…] + bias[…]))
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runMixedPipelineExample = runMixedPipelineExample;
const graph_1 = require("../ir/graph");
const passManager_1 = require("../passes/passManager");
const layoutTransformPass_1 = require("../passes/layoutTransformPass");
const layoutRules_1 = require("../patterns/layoutRules");
const rules_1 = require("../patterns/rules");
const opContracts_1 = require("../ops/opContracts");
const fusionAnalysis_1 = require("../analysis/fusionAnalysis");
const printer_1 = require("../debug/printer");
const loopPrinter_1 = require("../debug/loopPrinter");
const pipelines_1 = require("../passes/pipelines");
function runMixedPipelineExample() {
    (0, graph_1.resetCounters)();
    // ── Build graph ────────────────────────────────────────────────────────────
    const g = new graph_1.Graph();
    const x = g.addInputTensor("x_NCHW", "float32", [1, 3, 224, 224]);
    const bias = g.addInputTensor("bias", "float32", [1, 3, 224, 224]);
    const tp1 = g.addNode("transpose", [x.id], [{ name: "x_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }], { perm: [0, 2, 3, 1], fromLayout: "NCHW", toLayout: "NHWC" });
    const add = g.addNode("add", [tp1.outputs[0], bias.id], [{ name: "sum_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }]);
    const tp2 = g.addNode("transpose", [add.outputs[0]], [{ name: "sum_NCHW", dtype: "float32", shape: [1, 3, 224, 224] }], { perm: [0, 3, 1, 2], fromLayout: "NHWC", toLayout: "NCHW" });
    const relu = g.addNode("relu", [tp2.outputs[0]], [{ name: "out", dtype: "float32", shape: [1, 3, 224, 224] }]);
    g.markOutputs(relu.outputs[0]);
    const original = g.clone();
    // ── Intermediate: layout-clean graph for fusion diagnostics ───────────────
    const afterLayout = (() => {
        const singleStage = new passManager_1.PassManager({ validateAfterEachPass: true });
        singleStage.addPass(new layoutTransformPass_1.LayoutTransformPass(new layoutRules_1.LayoutRuleRegistry(), opContracts_1.DEFAULT_CONTRACT_REGISTRY));
        return singleStage.run(g.clone());
    })();
    const rules = new rules_1.RuleRegistry(rules_1.DEFAULT_FUSION_RULES);
    const fusionAnalysis = (0, fusionAnalysis_1.analyzeFusion)(afterLayout, rules.getRules(), opContracts_1.DEFAULT_CONTRACT_REGISTRY);
    (0, printer_1.printFusionAnalysis)(fusionAnalysis, "Fusion Analysis (after layout pass)");
    // ── Full 3-stage pipeline: Layout → Fusion → LoopLowering ────────────────
    const { pm, loopPass } = (0, pipelines_1.createDefaultPipeline)();
    const final = pm.run(original);
    (0, printer_1.printGraph)(original, "Mixed Pipeline — original");
    (0, printer_1.printGraph)(final, "Mixed Pipeline — after layout + fusion");
    (0, printer_1.printDiff)(original, final, "LayoutTransformPass + FusionPass + LoopLoweringPass");
    // ── Loop IR — the payoff: one fused nest, no intermediate buffers ─────────
    const loopModule = loopPass.getLastModule();
    if (loopModule) {
        (0, loopPrinter_1.printLoopModule)(loopModule, "Loop IR — fused add_relu kernel");
    }
}
//# sourceMappingURL=mixedPipeline.js.map