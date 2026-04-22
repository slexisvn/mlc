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

import { Graph, resetCounters } from "../ir/graph";
import { RuleRegistry }         from "../patterns/rules";
import { CostModel }            from "../optimizer/costModel";
import { FusionPass }           from "../passes/fusionPass";
import { PassManager }          from "../passes/passManager";
import { printGraph, printExecutionPlan, printDiff } from "../debug/printer";

export function runBranchingExample(): void {
  resetCounters();

  console.log("\n╔══════════════════════════════════════════════════════════╗");
  console.log("║  EXAMPLE 2 — Branching graph: fusion MUST be rejected   ║");
  console.log("╚══════════════════════════════════════════════════════════╝");

  // ── Build the graph ───────────────────────────────────────────────────────
  const g = new Graph("branching");

  const a = g.addInputTensor("a", "float32", [4, 4]);
  const b = g.addInputTensor("b", "float32", [4, 4]);

  // add: produces tensor c
  const addNode = g.addNode("add", [a.id, b.id], [{ name: "c", dtype: "float32", shape: [4, 4] }]);
  const c       = addNode.outputs[0];

  // relu:    consumes c  → output d
  const reluNode    = g.addNode("relu",    [c], [{ name: "d", dtype: "float32", shape: [4, 4] }]);
  // sigmoid: also consumes c  → output e
  const sigmoidNode = g.addNode("sigmoid", [c], [{ name: "e", dtype: "float32", shape: [4, 4] }]);

  // Both d and e are graph outputs.
  g.markOutputs(reluNode.outputs[0], sigmoidNode.outputs[0]);

  // ── Before ────────────────────────────────────────────────────────────────
  printGraph(g, "BEFORE optimisation");
  printExecutionPlan(g, "Execution plan BEFORE");

  console.log("  ⚠ Expecting: NO fusion (tensor `c` has 2 consumers)");

  // ── Optimise ──────────────────────────────────────────────────────────────
  const registry  = new RuleRegistry();
  const costModel = new CostModel();
  const pm        = new PassManager({ validateAfterEachPass: true });
  pm.addPass(new FusionPass(registry, costModel));

  const optimised = pm.run(g);

  // ── After ─────────────────────────────────────────────────────────────────
  printGraph(optimised, "AFTER optimisation");
  printExecutionPlan(optimised, "Execution plan AFTER");
  printDiff(g, optimised, "FusionPass");

  const fusionHappened = optimised.nodeOrder.length < g.nodeOrder.length;
  console.log(
    fusionHappened
      ? "  ✗ BUG: fusion happened when it should not have!"
      : "  ✓ Correct: branching graph was NOT fused.",
  );
}
