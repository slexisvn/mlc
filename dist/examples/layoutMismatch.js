"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/layoutMismatch.ts
//
// Demonstrates layout conflict detection.
//
// Graph:
//   x[NHWC] → conv_strict (requires NCHW only) → y [output]
//
// The LayoutTransformPass does NOT auto-resolve conflicts — it detects and
// reports them.  The graph is passed through unchanged, and the PassManager
// emits a warning so the developer knows a manual fix is needed.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runLayoutMismatchExample = runLayoutMismatchExample;
const graph_1 = require("../ir/graph");
const passManager_1 = require("../passes/passManager");
const layoutTransformPass_1 = require("../passes/layoutTransformPass");
const layoutRules_1 = require("../patterns/layoutRules");
const opContracts_1 = require("../ops/opContracts");
const layoutAnalysis_1 = require("../analysis/layoutAnalysis");
const printer_1 = require("../debug/printer");
function runLayoutMismatchExample() {
    (0, graph_1.resetCounters)();
    // ── Custom registry: a conv that only accepts NCHW ────────────────────────
    const strictRegistry = new opContracts_1.OpContractRegistry(opContracts_1.DEFAULT_OP_CONTRACTS);
    strictRegistry.register({
        op: "conv_strict",
        fusibilityClass: "fusible",
        layoutBehavior: "sensitive",
        requiredInputLayouts: ["NCHW"], // no NHWC allowed
        description: "Conv that strictly requires NCHW input.",
    });
    // ── Build graph with an NHWC input fed to a strict-NCHW conv ──────────────
    const g = new graph_1.Graph();
    const x = g.addInputTensor("x_NHWC", "float32", [1, 224, 224, 3]);
    const conv = g.addNode("conv_strict", [x.id], [{ name: "y", dtype: "float32", shape: [1, 64, 224, 224] }]);
    g.markOutputs(conv.outputs[0]);
    // ── Run layout analysis to surface the conflict ───────────────────────────
    const facts = (0, layoutAnalysis_1.analyzeLayouts)(g, strictRegistry);
    (0, printer_1.printLayoutAnalysis)(facts, "Layout Mismatch — conflict report");
    (0, printer_1.printGraph)(g, "Layout Mismatch — graph (unchanged)");
    // ── Run LayoutTransformPass — it warns but does not crash ─────────────────
    const pm = new passManager_1.PassManager({ validateAfterEachPass: true });
    pm.addPass(new layoutTransformPass_1.LayoutTransformPass(new layoutRules_1.LayoutRuleRegistry(), strictRegistry));
    pm.run(g); // graph unchanged; conflicts logged
    console.log("  Note: the graph was not rewritten — a manual transpose insertion");
    console.log("        is required to satisfy the layout contract of conv_strict.\n");
}
//# sourceMappingURL=layoutMismatch.js.map