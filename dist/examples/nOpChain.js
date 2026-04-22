"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/nOpChain.ts
//
// BONUS Example 3: Multi-op (N-op) chain fusion.
//
// Graph:
//   x, w  ──►  matmul  ──►  z  ──►  add  ──►  h  ──►  relu  ──►  y  [output]
//                                     ▲
//                                     b (bias tensor, graph input)
//
// Rule matched: ["matmul", "add", "relu"] → "linear_relu"  (3-op fusion)
//
// Node count: 3 → 1
//
// This demonstrates that the chain matcher generalises cleanly to N ops
// without any special-case code — longer rules simply take priority in the
// sorted rule list.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runNOpChainExample = runNOpChainExample;
const graph_1 = require("../ir/graph");
const rules_1 = require("../patterns/rules");
const costModel_1 = require("../optimizer/costModel");
const fusionPass_1 = require("../passes/fusionPass");
const loopLoweringPass_1 = require("../passes/loopLoweringPass");
const passManager_1 = require("../passes/passManager");
const printer_1 = require("../debug/printer");
const loopPrinter_1 = require("../debug/loopPrinter");
function runNOpChainExample() {
    (0, graph_1.resetCounters)();
    console.log("\n╔══════════════════════════════════════════════════════════╗");
    console.log("║  EXAMPLE 3 — 3-op chain: matmul → add → relu            ║");
    console.log("╚══════════════════════════════════════════════════════════╝");
    // ── Build the graph ───────────────────────────────────────────────────────
    const g = new graph_1.Graph("n_op_chain");
    const x = g.addInputTensor("x", "float32", [8, 16]); // input activations
    const w = g.addInputTensor("w", "float32", [16, 32]); // weight matrix
    const b = g.addInputTensor("b", "float32", [32]); // bias vector
    // matmul: x @ w → z  [shape 8×32]
    const matmulNode = g.addNode("matmul", [x.id, w.id], [{ name: "z", dtype: "float32", shape: [8, 32] }]);
    // add: z + b → h  (bias addition)
    const addNode = g.addNode("add", [matmulNode.outputs[0], b.id], [{ name: "h", dtype: "float32", shape: [8, 32] }]);
    // relu: h → y
    const reluNode = g.addNode("relu", [addNode.outputs[0]], [{ name: "y", dtype: "float32", shape: [8, 32] }]);
    g.markOutputs(reluNode.outputs[0]);
    // ── Before ────────────────────────────────────────────────────────────────
    (0, printer_1.printGraph)(g, "BEFORE optimisation");
    (0, printer_1.printExecutionPlan)(g, "Execution plan BEFORE");
    // ── Optimise ──────────────────────────────────────────────────────────────
    const registry = new rules_1.RuleRegistry(); // includes ["matmul","add","relu"] → "linear_relu"
    const costModel = new costModel_1.CostModel();
    const loopPass = new loopLoweringPass_1.LoopLoweringPass();
    const pm = new passManager_1.PassManager({ validateAfterEachPass: true });
    pm.addPasses(new fusionPass_1.FusionPass(registry, costModel), loopPass);
    const optimised = pm.run(g);
    // ── After ─────────────────────────────────────────────────────────────────
    (0, printer_1.printGraph)(optimised, "AFTER optimisation");
    (0, printer_1.printExecutionPlan)(optimised, "Execution plan AFTER");
    (0, printer_1.printDiff)(g, optimised, "FusionPass");
    // ── Loop IR — fused linear_relu: matmul + bias + relu in one nest ─────────
    const loopModule = loopPass.getLastModule();
    if (loopModule) {
        (0, loopPrinter_1.printLoopModule)(loopModule, "Loop IR — fused linear_relu kernel");
    }
}
//# sourceMappingURL=nOpChain.js.map