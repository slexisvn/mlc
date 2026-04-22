"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/loopTiling.ts
//
// Demo: loop tiling (strip-mining) of a matmul loop nest.
//
// Graph:  x[M, K]  ×  w[K, N]  →  matmul  →  out[M, N]
//
// After LoopLoweringPass the LoopFunction body is the canonical matmul nest:
//   for i in [0, M):
//     for j in [0, N):
//       out[i, j] = 0.0
//       for k in [0, K):
//         out[i, j] += x[i, k] * w[k, j]
//
// After LoopTilingPass (T=32, minBound=32, tileReductions=false):
//   for i_o in [0, ⌈M/32⌉):
//     for i_i in [0, 32):            ← or min(32, M − i_o·32) for edge tiles
//       for j_o in [0, ⌈N/32⌉):
//         for j_i in [0, 32):        ← edge-tile handling included
//           out[...] = 0.0
//           for k in [0, K):         ← reduction dimension NOT tiled (default)
//             out[...] += x[...] * w[...]
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runLoopTilingExample = runLoopTilingExample;
const graph_1 = require("../ir/graph");
const loopLoweringPass_1 = require("../passes/loopLoweringPass");
const passManager_1 = require("../passes/passManager");
const loopPass_1 = require("../passes/loopPass");
const loopTilingPass_1 = require("../passes/loopTilingPass");
const loopPrinter_1 = require("../debug/loopPrinter");
const printer_1 = require("../debug/printer");
function runLoopTilingExample() {
    (0, graph_1.resetCounters)();
    // ── Build a matmul graph ──────────────────────────────────────────────────
    const M = 128, K = 64, N = 96;
    const g = new graph_1.Graph();
    const x = g.addInputTensor("x", "float32", [M, K]);
    const w = g.addInputTensor("w", "float32", [K, N]);
    const mm = g.addNode("matmul", [x.id, w.id], [{ name: "out", dtype: "float32", shape: [M, N] }]);
    g.markOutputs(mm.outputs[0]);
    (0, printer_1.printGraph)(g, "Loop Tiling Example — matmul graph");
    // ── Lower to Loop IR ──────────────────────────────────────────────────────
    const lowerPass = new loopLoweringPass_1.LoopLoweringPass();
    const pm = new passManager_1.PassManager({ validateAfterEachPass: false });
    pm.addPass(lowerPass);
    pm.run(g);
    const beforeModule = lowerPass.getLastModule();
    (0, loopPrinter_1.printLoopModule)(beforeModule, "Loop IR — BEFORE tiling (canonical matmul nest)");
    // ── Apply LoopTilingPass ──────────────────────────────────────────────────
    // Tile size 32 for all dimensions; minBound=32 so all of i, j qualify.
    // k-loop is a reduction → not tiled by default.
    const loopPm = new loopPass_1.LoopPassManager({ validateAfterEachPass: true });
    loopPm.addPass(new loopTilingPass_1.LoopTilingPass({ defaultTileSize: 32, minBound: 32 }));
    const afterModule = loopPm.run(beforeModule);
    (0, loopPrinter_1.printLoopModule)(afterModule, "Loop IR — AFTER tiling (i and j strip-mined, k unchanged)");
    // Verify tile structure: count nesting depth of the outermost ForLoop chain.
    const fn = afterModule.functions[0];
    let depth = 0;
    let cur = fn.body[0];
    while (cur && cur.kind === "ForLoop") {
        depth++;
        cur = cur.body[0];
    }
    console.log(`  ✓ Tiling result: outermost loop nesting depth = ${depth}` +
        ` (expected >= 4: i_o, i_i, j_o, j_i at minimum).`);
}
//# sourceMappingURL=loopTiling.js.map