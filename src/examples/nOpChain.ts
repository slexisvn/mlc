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

import { Graph, resetCounters } from "../ir/graph";
import { RuleRegistry }         from "../patterns/rules";
import { CostModel }            from "../optimizer/costModel";
import { FusionPass }           from "../passes/fusionPass";
import { LoopLoweringPass }     from "../passes/loopLoweringPass";
import { PassManager }          from "../passes/passManager";
import { printGraph, printExecutionPlan, printDiff } from "../debug/printer";
import { printLoopModule }      from "../debug/loopPrinter";

export function runNOpChainExample(): void {
  resetCounters();

  console.log("\n╔══════════════════════════════════════════════════════════╗");
  console.log("║  EXAMPLE 3 — 3-op chain: matmul → add → relu            ║");
  console.log("╚══════════════════════════════════════════════════════════╝");

  // ── Build the graph ───────────────────────────────────────────────────────
  const g = new Graph("n_op_chain");

  const x = g.addInputTensor("x", "float32", [8, 16]);   // input activations
  const w = g.addInputTensor("w", "float32", [16, 32]);  // weight matrix
  const b = g.addInputTensor("b", "float32", [32]);      // bias vector

  // matmul: x @ w → z  [shape 8×32]
  const matmulNode = g.addNode(
    "matmul",
    [x.id, w.id],
    [{ name: "z", dtype: "float32", shape: [8, 32] }],
  );

  // add: z + b → h  (bias addition)
  const addNode = g.addNode(
    "add",
    [matmulNode.outputs[0], b.id],
    [{ name: "h", dtype: "float32", shape: [8, 32] }],
  );

  // relu: h → y
  const reluNode = g.addNode(
    "relu",
    [addNode.outputs[0]],
    [{ name: "y", dtype: "float32", shape: [8, 32] }],
  );

  g.markOutputs(reluNode.outputs[0]);

  // ── Before ────────────────────────────────────────────────────────────────
  printGraph(g, "BEFORE optimisation");
  printExecutionPlan(g, "Execution plan BEFORE");

  // ── Optimise ──────────────────────────────────────────────────────────────
  const registry  = new RuleRegistry();    // includes ["matmul","add","relu"] → "linear_relu"
  const costModel = new CostModel();
  const loopPass  = new LoopLoweringPass();
  const pm        = new PassManager({ validateAfterEachPass: true });
  pm.addPasses(new FusionPass(registry, costModel), loopPass);

  const optimised = pm.run(g);

  // ── After ─────────────────────────────────────────────────────────────────
  printGraph(optimised, "AFTER optimisation");
  printExecutionPlan(optimised, "Execution plan AFTER");
  printDiff(g, optimised, "FusionPass");

  // ── Loop IR — fused linear_relu: matmul + bias + relu in one nest ─────────
  const loopModule = loopPass.getLastModule();
  if (loopModule) {
    printLoopModule(loopModule, "Loop IR — fused linear_relu kernel");
  }
}
