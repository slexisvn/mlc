"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/loopOptimization.ts
//
// End-to-end demo: full two-stage pipeline (graph + loop IR).
//
// Graph:
//   x[128, 64]  ×  w[64, 96]  +  bias[96]  →  linear_relu  →  out[128, 96]
//     ↑ graph-level FusionPass fuses matmul + bias + relu into linear_relu
//
//   The graph pipeline (LayoutTransform → Fusion → LoopLowering) then produces
//   a single LoopFunction with the fused linear_relu nest.
//
//   The loop pipeline (LoopFusion → LoopTiling) then:
//     • LoopFusionPass: the fused linear_relu nest is non-perfect (mixed body),
//       so no loop-level fusion fires — logged as a rejection.
//     • LoopTilingPass: tiles the i and j dimensions of the linear_relu nest,
//       leaving the reduction k-loop unchanged.
//
// This demonstrates:
//   1. Graph-level and loop-level optimisation as independent stages.
//   2. Correct pass ordering: fusion (graph) → lowering → loop fusion → tiling.
//   3. Conservative safety: non-perfect nests are not incorrectly fused.
//   4. End-to-end observability via printLoopModule().
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runLoopOptimizationExample = runLoopOptimizationExample;
const graph_1 = require("../ir/graph");
const printer_1 = require("../debug/printer");
const loopPrinter_1 = require("../debug/loopPrinter");
const pipelines_1 = require("../passes/pipelines");
function runLoopOptimizationExample() {
    (0, graph_1.resetCounters)();
    // ── Build graph: matmul + bias-add + relu (pre-fusion shape) ─────────────
    const M = 128, K = 64, N = 96;
    const g = new graph_1.Graph();
    const x = g.addInputTensor("x", "float32", [M, K]);
    const w = g.addInputTensor("w", "float32", [K, N]);
    const bias = g.addInputTensor("bias", "float32", [N]);
    const mm = g.addNode("matmul", [x.id, w.id], [{ name: "mm_out", dtype: "float32", shape: [M, N] }]);
    const add = g.addNode("add", [mm.outputs[0], bias.id], [{ name: "add_out", dtype: "float32", shape: [M, N] }]);
    const relu = g.addNode("relu", [add.outputs[0]], [{ name: "out", dtype: "float32", shape: [M, N] }]);
    g.markOutputs(relu.outputs[0]);
    const original = g.clone();
    (0, printer_1.printGraph)(original, "Loop Optimization — original graph (matmul + add + relu)");
    // ── Stage 1: graph pipeline ───────────────────────────────────────────────
    // Order: LayoutTransformPass → FusionPass → LoopLoweringPass.
    // FusionPass should fuse matmul + add + relu into linear_relu.
    const { pm, loopPass, loopPm } = (0, pipelines_1.createFullPipeline)({ validateAfterEachPass: true }, {
        tiling: {
            defaultTileSize: 32,
            minBound: 32,
            tileReductions: false,
        },
    });
    const graphOptimized = pm.run(g);
    (0, printer_1.printGraph)(graphOptimized, "Loop Optimization — after graph pipeline (linear_relu fused)");
    (0, printer_1.printDiff)(original, graphOptimized, "Graph pipeline delta");
    const rawModule = loopPass.getLastModule();
    (0, loopPrinter_1.printLoopModule)(rawModule, "Loop IR — after lowering (fused linear_relu nest)");
    // ── Stage 2: loop pipeline ────────────────────────────────────────────────
    // Order: LoopFusionPass (no-op for non-perfect nests) → LoopTilingPass.
    const optimizedModule = loopPm.run(rawModule);
    (0, loopPrinter_1.printLoopModule)(optimizedModule, "Loop IR — after loop optimization (tiled i and j dims)");
    // ── Summary ───────────────────────────────────────────────────────────────
    const fnBefore = rawModule.functions[0];
    const fnAfter = optimizedModule.functions[0];
    let depthBefore = 0;
    let depthAfter = 0;
    let cur = fnBefore.body[0];
    while (cur && cur.kind === "ForLoop") {
        depthBefore++;
        cur = cur.body[0];
    }
    cur = fnAfter.body[0];
    while (cur && cur.kind === "ForLoop") {
        depthAfter++;
        cur = cur.body[0];
    }
    console.log(`\n  Summary: "${fnBefore.name}"`);
    console.log(`    Loop nesting depth before optimization : ${depthBefore}`);
    console.log(`    Loop nesting depth after  optimization : ${depthAfter}`);
    console.log(`    ✓ Tiling added ${depthAfter - depthBefore} extra nesting level(s).`);
}
//# sourceMappingURL=loopOptimization.js.map