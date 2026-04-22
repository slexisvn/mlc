"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// examples/branching.ts
//
// Example 2: Branching graph — fusion must be REJECTED.
//
// Graph:
//   a, b  ──►  add  ──►  c  ──►  relu     ──►  d  [output1]
//                         └──►  sigmoid  ──►  e  [output2]
//
// The tensor `c` (add's output) has TWO consumers: relu and sigmoid.
// Invariant: intermediate tensors must have exactly one consumer.
// Therefore the matcher rejects both add→relu and add→sigmoid chains.
//
// Expected result after FusionPass: graph UNCHANGED.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.runBranchingExample = runBranchingExample;
const graph_1 = require("../ir/graph");
const rules_1 = require("../patterns/rules");
const costModel_1 = require("../optimizer/costModel");
const fusionPass_1 = require("../passes/fusionPass");
const passManager_1 = require("../passes/passManager");
const printer_1 = require("../debug/printer");
function runBranchingExample() {
    (0, graph_1.resetCounters)();
    console.log("\n╔══════════════════════════════════════════════════════════╗");
    console.log("║  EXAMPLE 2 — Branching graph: fusion MUST be rejected   ║");
    console.log("╚══════════════════════════════════════════════════════════╝");
    // ── Build the graph ───────────────────────────────────────────────────────
    const g = new graph_1.Graph("branching");
    const a = g.addInputTensor("a", "float32", [4, 4]);
    const b = g.addInputTensor("b", "float32", [4, 4]);
    // add: produces tensor c
    const addNode = g.addNode("add", [a.id, b.id], [{ name: "c", dtype: "float32", shape: [4, 4] }]);
    const c = addNode.outputs[0];
    // relu:    consumes c  → output d
    const reluNode = g.addNode("relu", [c], [{ name: "d", dtype: "float32", shape: [4, 4] }]);
    // sigmoid: also consumes c  → output e
    const sigmoidNode = g.addNode("sigmoid", [c], [{ name: "e", dtype: "float32", shape: [4, 4] }]);
    // Both d and e are graph outputs.
    g.markOutputs(reluNode.outputs[0], sigmoidNode.outputs[0]);
    // ── Before ────────────────────────────────────────────────────────────────
    (0, printer_1.printGraph)(g, "BEFORE optimisation");
    (0, printer_1.printExecutionPlan)(g, "Execution plan BEFORE");
    console.log("  ⚠ Expecting: NO fusion (tensor `c` has 2 consumers)");
    // ── Optimise ──────────────────────────────────────────────────────────────
    const registry = new rules_1.RuleRegistry();
    const costModel = new costModel_1.CostModel();
    const pm = new passManager_1.PassManager({ validateAfterEachPass: true });
    pm.addPass(new fusionPass_1.FusionPass(registry, costModel));
    const optimised = pm.run(g);
    // ── After ─────────────────────────────────────────────────────────────────
    (0, printer_1.printGraph)(optimised, "AFTER optimisation");
    (0, printer_1.printExecutionPlan)(optimised, "Execution plan AFTER");
    (0, printer_1.printDiff)(g, optimised, "FusionPass");
    const fusionHappened = optimised.nodeOrder.length < g.nodeOrder.length;
    console.log(fusionHappened
        ? "  ✗ BUG: fusion happened when it should not have!"
        : "  ✓ Correct: branching graph was NOT fused.");
}
//# sourceMappingURL=branching.js.map