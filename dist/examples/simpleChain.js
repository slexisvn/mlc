"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/simpleChain.ts
//
// Example 1: Simple fusible linear chain.
//
// Graph:
//   a, b  ──►  add  ──►  c  ──►  relu  ──►  d  [output]
//
// Expected result after FusionPass:
//   a, b  ──►  add_relu  ──►  d_fused  [output]
//
// Node count: 2 → 1
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runSimpleChainExample = runSimpleChainExample;
const graph_1 = require("../ir/graph");
const rules_1 = require("../patterns/rules");
const costModel_1 = require("../optimizer/costModel");
const fusionPass_1 = require("../passes/fusionPass");
const loopLoweringPass_1 = require("../passes/loopLoweringPass");
const passManager_1 = require("../passes/passManager");
const printer_1 = require("../debug/printer");
const loopPrinter_1 = require("../debug/loopPrinter");
function runSimpleChainExample() {
    (0, graph_1.resetCounters)();
    console.log("\n╔══════════════════════════════════════════════════════════╗");
    console.log("║  EXAMPLE 1 — Simple fusible chain: add → relu           ║");
    console.log("╚══════════════════════════════════════════════════════════╝");
    // ── Build the graph ───────────────────────────────────────────────────────
    const g = new graph_1.Graph("simple_chain");
    const a = g.addInputTensor("a", "float32", [4, 4]);
    const b = g.addInputTensor("b", "float32", [4, 4]);
    const addNode = g.addNode("add", [a.id, b.id], [{ name: "c", dtype: "float32", shape: [4, 4] }]);
    const reluNode = g.addNode("relu", [addNode.outputs[0]], [{ name: "d", dtype: "float32", shape: [4, 4] }]);
    g.markOutputs(reluNode.outputs[0]);
    // ── Before ────────────────────────────────────────────────────────────────
    (0, printer_1.printGraph)(g, "BEFORE optimisation");
    (0, printer_1.printExecutionPlan)(g, "Execution plan BEFORE");
    // ── Optimise ──────────────────────────────────────────────────────────────
    const registry = new rules_1.RuleRegistry();
    const costModel = new costModel_1.CostModel();
    const loopPass = new loopLoweringPass_1.LoopLoweringPass();
    const pm = new passManager_1.PassManager({ validateAfterEachPass: true });
    pm.addPasses(new fusionPass_1.FusionPass(registry, costModel), loopPass);
    const optimised = pm.run(g);
    // ── After ─────────────────────────────────────────────────────────────────
    (0, printer_1.printGraph)(optimised, "AFTER optimisation");
    (0, printer_1.printExecutionPlan)(optimised, "Execution plan AFTER");
    (0, printer_1.printDiff)(g, optimised, "FusionPass");
    // ── Loop IR ───────────────────────────────────────────────────────────────
    const loopModule = loopPass.getLastModule();
    if (loopModule) {
        (0, loopPrinter_1.printLoopModule)(loopModule, "Loop IR — fused add_relu kernel");
    }
}
//# sourceMappingURL=simpleChain.js.map