"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/layoutPropagation.ts
//
// Demonstrates transpose sandwich removal around a layout-agnostic add.
//
// Graph (before):
//   x[NCHW] → transpose(NCHW→NHWC) → x_nhwc
//   x_nhwc, bias → add → sum_NHWC
//   sum_NHWC → transpose(NHWC→NCHW) → result [output]
//
// The add op is layout-agnostic, so both transposes are unnecessary.
//
// After LayoutTransformPass:
//   x[NCHW], bias → add → sum_NHWC [output]
//   (the add now operates in NCHW layout; both transposes eliminated)
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runLayoutPropagationExample = runLayoutPropagationExample;
const graph_1 = require("../ir/graph");
const passManager_1 = require("../passes/passManager");
const layoutTransformPass_1 = require("../passes/layoutTransformPass");
const layoutRules_1 = require("../patterns/layoutRules");
const opContracts_1 = require("../ops/opContracts");
const printer_1 = require("../debug/printer");
function runLayoutPropagationExample() {
    (0, graph_1.resetCounters)();
    // ── Build graph ────────────────────────────────────────────────────────────
    const g = new graph_1.Graph();
    const x = g.addInputTensor("x_NCHW", "float32", [1, 3, 224, 224]);
    const bias = g.addInputTensor("bias", "float32", [1, 3, 224, 224]);
    // NCHW → NHWC
    const tp1 = g.addNode("transpose", [x.id], [{ name: "x_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }], { perm: [0, 2, 3, 1], fromLayout: "NCHW", toLayout: "NHWC" });
    // add(x_nhwc, bias) — layout-agnostic
    const add = g.addNode("add", [tp1.outputs[0], bias.id], [{ name: "sum_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }]);
    // NHWC → NCHW  (inverse of tp1)
    const tp2 = g.addNode("transpose", [add.outputs[0]], [{ name: "result_NCHW", dtype: "float32", shape: [1, 3, 224, 224] }], { perm: [0, 3, 1, 2], fromLayout: "NHWC", toLayout: "NCHW" });
    g.markOutputs(tp2.outputs[0]);
    // ── Run LayoutTransformPass ───────────────────────────────────────────────
    const before = g.clone();
    const pm = new passManager_1.PassManager({ validateAfterEachPass: true });
    pm.addPass(new layoutTransformPass_1.LayoutTransformPass(new layoutRules_1.LayoutRuleRegistry(), opContracts_1.DEFAULT_CONTRACT_REGISTRY));
    const after = pm.run(g);
    (0, printer_1.printGraph)(before, "Layout Propagation — before");
    (0, printer_1.printGraph)(after, "Layout Propagation — after");
    (0, printer_1.printDiff)(before, after, "LayoutTransformPass (propagation)");
}
//# sourceMappingURL=layoutPropagationExample.js.map