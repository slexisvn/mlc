"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/layoutCancellation.ts
//
// Demonstrates direct transpose–transpose cancellation.
//
// Graph (before):
//   x[NCHW] → transpose(NCHW→NHWC) → x_nhwc
//           → transpose(NHWC→NCHW) → x_back [output]
//
// After LayoutTransformPass:
//   x[NCHW] → x_back (=x, both transposes eliminated)
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runLayoutCancellationExample = runLayoutCancellationExample;
const graph_1 = require("../ir/graph");
const passManager_1 = require("../passes/passManager");
const layoutTransformPass_1 = require("../passes/layoutTransformPass");
const layoutRules_1 = require("../patterns/layoutRules");
const opContracts_1 = require("../ops/opContracts");
const layoutAnalysis_1 = require("../analysis/layoutAnalysis");
const printer_1 = require("../debug/printer");
function runLayoutCancellationExample() {
    (0, graph_1.resetCounters)();
    // ── Build graph ────────────────────────────────────────────────────────────
    const g = new graph_1.Graph();
    const x = g.addInputTensor("x_NCHW", "float32", [1, 3, 224, 224]);
    // NCHW → NHWC  (perm = [0, 2, 3, 1])
    const tp1 = g.addNode("transpose", [x.id], [{ name: "x_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }], { perm: [0, 2, 3, 1], fromLayout: "NCHW", toLayout: "NHWC" });
    // NHWC → NCHW  (perm = [0, 3, 1, 2]) — inverse of tp1
    const tp2 = g.addNode("transpose", [tp1.outputs[0]], [{ name: "x_back_NCHW", dtype: "float32", shape: [1, 3, 224, 224] }], { perm: [0, 3, 1, 2], fromLayout: "NHWC", toLayout: "NCHW" });
    g.markOutputs(tp2.outputs[0]);
    // ── Run layout analysis first (informational) ─────────────────────────────
    const layoutFacts = (0, layoutAnalysis_1.analyzeLayouts)(g, opContracts_1.DEFAULT_CONTRACT_REGISTRY);
    (0, printer_1.printLayoutAnalysis)(layoutFacts, "Layout Analysis — before cancellation");
    // ── Run LayoutTransformPass ───────────────────────────────────────────────
    const before = g.clone();
    const pm = new passManager_1.PassManager({ validateAfterEachPass: true });
    pm.addPass(new layoutTransformPass_1.LayoutTransformPass(new layoutRules_1.LayoutRuleRegistry(), opContracts_1.DEFAULT_CONTRACT_REGISTRY));
    const after = pm.run(g);
    (0, printer_1.printGraph)(before, "Layout Cancellation — before");
    (0, printer_1.printGraph)(after, "Layout Cancellation — after");
    (0, printer_1.printDiff)(before, after, "LayoutTransformPass (cancellation)");
}
//# sourceMappingURL=layoutCancellation.js.map