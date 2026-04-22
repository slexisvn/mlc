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

import { Graph, resetCounters } from "../ir/graph";
import { RuleRegistry }         from "../patterns/rules";
import { CostModel }            from "../optimizer/costModel";
import { FusionPass }           from "../passes/fusionPass";
import { LoopLoweringPass }     from "../passes/loopLoweringPass";
import { PassManager }          from "../passes/passManager";
import { printGraph, printExecutionPlan, printDiff } from "../debug/printer";
import { printLoopModule }      from "../debug/loopPrinter";

export function runSimpleChainExample(): void {
  resetCounters();

  console.log("\n╔══════════════════════════════════════════════════════════╗");
  console.log("║  EXAMPLE 1 — Simple fusible chain: add → relu           ║");
  console.log("╚══════════════════════════════════════════════════════════╝");

  // ── Build the graph ───────────────────────────────────────────────────────
  const g = new Graph("simple_chain");

  const a = g.addInputTensor("a", "float32", [4, 4]);
  const b = g.addInputTensor("b", "float32", [4, 4]);

  const addNode  = g.addNode("add",  [a.id, b.id], [{ name: "c", dtype: "float32", shape: [4, 4] }]);
  const reluNode = g.addNode("relu", [addNode.outputs[0]], [{ name: "d", dtype: "float32", shape: [4, 4] }]);

  g.markOutputs(reluNode.outputs[0]);

  // ── Before ────────────────────────────────────────────────────────────────
  printGraph(g, "BEFORE optimisation");
  printExecutionPlan(g, "Execution plan BEFORE");

  // ── Optimise ──────────────────────────────────────────────────────────────
  const registry   = new RuleRegistry();
  const costModel  = new CostModel();
  const loopPass   = new LoopLoweringPass();
  const pm         = new PassManager({ validateAfterEachPass: true });
  pm.addPasses(new FusionPass(registry, costModel), loopPass);

  const optimised = pm.run(g);

  // ── After ─────────────────────────────────────────────────────────────────
  printGraph(optimised, "AFTER optimisation");
  printExecutionPlan(optimised, "Execution plan AFTER");
  printDiff(g, optimised, "FusionPass");

  // ── Loop IR ───────────────────────────────────────────────────────────────
  const loopModule = loopPass.getLastModule();
  if (loopModule) {
    printLoopModule(loopModule, "Loop IR — fused add_relu kernel");
  }
}
