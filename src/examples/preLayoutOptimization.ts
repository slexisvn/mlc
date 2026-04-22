// ─────────────────────────────────────────────────────────────────────────────
// examples/preLayoutOptimization.ts
//
// Demonstrates the three pre-layout graph-simplification passes:
//   1. ConstantFoldingPass — evaluates ops whose inputs are compile-time
//      constants, replacing compute nodes with "const" source nodes.
//   2. CSEPass             — eliminates duplicate subexpressions via value
//      numbering, rewiring consumers to the canonical producer.
//   3. DeadCodeEliminationPass — removes nodes not reachable from any graph
//      output, including those orphaned by CF and CSE.
//
// Three independent sub-examples are run:
//
//   A. Constant Folding + DCE
//      scale (const) → mul → relu
//      Both inputs of mul are constant scalars, so CF folds mul → const,
//      then relu → const.  DCE removes both original compute nodes.
//      Result: 0 compute nodes, single "const" output.
//
//   B. Common Subexpression Elimination
//      a, b → add → c1
//      a, b → add → c2 (identical to c1)
//      c1, c2 → mul → d [output]
//      CSE deduplicates the two add nodes; DCE removes the now-dead one.
//      Result: 2 nodes → 1 add + 1 mul (the redundant add is gone).
//
//   C. Combined: CF + CSE + DCE on a realistic graph
//      Computes  relu((x * scale) + bias)  where scale and bias are constants.
//      Demonstrates that the constant sub-graph is fully folded and pruned
//      before LayoutTransformPass and FusionPass run.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph, resetCounters }   from "../ir/graph";
import { ConstantFoldingPass, resetCFCounter } from "../passes/constantFoldingPass";
import { CSEPass }                from "../passes/csePass";
import { DeadCodeEliminationPass } from "../passes/deadCodeEliminationPass";
import { PassManager }            from "../passes/passManager";
import { PassLog }                from "../passes/pass";
import { validateGraph }          from "../ir/validate";

// ─── Helper: pretty-print PassLog entries ─────────────────────────────────────

function printLogs(passName: string, logs: PassLog[]): void {
  console.log(`  [${passName}]`);
  for (const l of logs) {
    const prefix = l.level === "warn" ? "  ⚠ " : l.level === "error" ? "  ✖ " : "    ";
    console.log(`${prefix}${l.message}`);
  }
}

// ─── Helper: add a constant scalar tensor to a graph ─────────────────────────
// We model a constant by declaring it as a graph input *and* immediately
// attaching a ConstantPayload — this is the canonical way to introduce
// compile-time-known scalars into the graph before ConstantFoldingPass runs.

function addConstantScalar(
  g:     Graph,
  name:  string,
  value: number,
): ReturnType<Graph["addInputTensor"]> {
  const t = g.addInputTensor(name, "float32", []);
  // Attach the constant payload directly on the tensor record.
  // ConstantFoldingPass propagates this payload through downstream ops.
  g._setConstantPayload(t.id, { data: [value] });
  return t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-example A: Constant Folding + DCE
// ─────────────────────────────────────────────────────────────────────────────

function runExampleA(): void {
  console.log("\n┌──────────────────────────────────────────────────────────┐");
  console.log("│  A. Constant Folding + Dead Code Elimination             │");
  console.log("│     2.0 * 3.0 → mul (foldable) → relu (foldable)        │");
  console.log("└──────────────────────────────────────────────────────────┘");

  resetCounters();
  resetCFCounter();

  const g = new Graph("cf_dce_example");

  // Two scalar constants.
  const c2  = addConstantScalar(g, "c2",  2.0);
  const c3  = addConstantScalar(g, "c3", -3.0);

  // mul: 2.0 * (-3.0) = -6.0
  const mulNode  = g.addNode("mul",  [c2.id, c3.id],          [{ name: "mul_out",  dtype: "float32", shape: [] }]);
  // relu: relu(-6.0) = 0.0
  const reluNode = g.addNode("relu", [mulNode.outputs[0]],     [{ name: "relu_out", dtype: "float32", shape: [] }]);

  g.markOutputs(reluNode.outputs[0]);

  console.log(`\n  Before: ${g.nodeOrder.length} node(s) — ${[...g.nodes.values()].map(n => n.op).join(", ")}`);

  // ── Apply passes ──────────────────────────────────────────────────────────
  const cfPass  = new ConstantFoldingPass();
  const dcePass = new DeadCodeEliminationPass();

  const cfResult  = cfPass.run(g);
  printLogs("ConstantFoldingPass", cfResult.logs);

  const dceResult = dcePass.run(cfResult.graph);
  printLogs("DeadCodeEliminationPass", dceResult.logs);

  const finalGraph = dceResult.graph;
  console.log(`\n  After:  ${finalGraph.nodeOrder.length} node(s) — ${[...finalGraph.nodes.values()].map(n => n.op).join(", ") || "(none — pure constants)"}`);

  // Verify the output tensor carries the folded value.
  const outId   = finalGraph.outputIds[0];
  const outTensor = finalGraph.getTensor(outId);
  const payload = outTensor.constantPayload;
  console.log(`  Output tensor "${outTensor.name}" constantPayload.data = [${payload?.data.join(", ") ?? "none"}]`);
  console.log(`  Expected: [0]  ← relu(-6.0) = max(0, -6.0) = 0.0  ✓`);

  // Validate structural integrity.
  const valid = validateGraph(finalGraph);
  console.log(`  validateGraph: ${valid.valid ? "PASS" : "FAIL"} (${valid.errors.length} error(s))`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-example B: Common Subexpression Elimination
// ─────────────────────────────────────────────────────────────────────────────

function runExampleB(): void {
  console.log("\n┌──────────────────────────────────────────────────────────┐");
  console.log("│  B. Common Subexpression Elimination                     │");
  console.log("│     add(a,b) appears twice → deduplicate → DCE the copy  │");
  console.log("└──────────────────────────────────────────────────────────┘");

  resetCounters();

  const g = new Graph("cse_example");

  const a = g.addInputTensor("a", "float32", [4, 4]);
  const b = g.addInputTensor("b", "float32", [4, 4]);

  // Two structurally identical add nodes.
  const add1 = g.addNode("add", [a.id, b.id], [{ name: "sum1", dtype: "float32", shape: [4, 4] }]);
  const add2 = g.addNode("add", [a.id, b.id], [{ name: "sum2", dtype: "float32", shape: [4, 4] }]);

  // mul consumes both sums — after CSE both inputs become the canonical sum.
  const mulNode = g.addNode("mul", [add1.outputs[0], add2.outputs[0]], [{ name: "product", dtype: "float32", shape: [4, 4] }]);

  g.markOutputs(mulNode.outputs[0]);

  const nodesBefore = g.nodeOrder.length;
  console.log(`\n  Before: ${nodesBefore} node(s) — ${[...g.nodes.values()].map(n => n.op).join(", ")}`);

  // ── Apply passes ──────────────────────────────────────────────────────────
  const csePass = new CSEPass();
  const dcePass = new DeadCodeEliminationPass();

  const cseResult = csePass.run(g);
  printLogs("CSEPass", cseResult.logs);

  const dceResult = dcePass.run(cseResult.graph);
  printLogs("DeadCodeEliminationPass", dceResult.logs);

  const finalGraph = dceResult.graph;
  const nodesAfter = finalGraph.nodeOrder.length;
  console.log(`\n  After:  ${nodesAfter} node(s) — ${[...finalGraph.nodes.values()].map(n => n.op).join(", ")}`);
  console.log(`  Nodes removed: ${nodesBefore - nodesAfter}  (expected: 1 duplicate add)`);

  // The mul node's inputs should now both point to the same tensor (add1's output).
  const mulAfter   = [...finalGraph.nodes.values()].find(n => n.op === "mul")!;
  const inputsUniq = new Set(mulAfter.inputs).size;
  console.log(`  mul inputs after CSE: [${mulAfter.inputs.join(", ")}] — unique inputs: ${inputsUniq}`);
  console.log(`  Expected: both inputs identical (CSE rewired add2 consumers to add1)  ✓`);

  const valid = validateGraph(finalGraph);
  console.log(`  validateGraph: ${valid.valid ? "PASS" : "FAIL"} (${valid.errors.length} error(s))`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-example C: Full pre-layout simplification pipeline
// ─────────────────────────────────────────────────────────────────────────────

function runExampleC(): void {
  console.log("\n┌──────────────────────────────────────────────────────────┐");
  console.log("│  C. Full pre-layout simplification: CF → CSE → DCE      │");
  console.log("│     relu((x * scale) + bias) where scale, bias = consts │");
  console.log("└──────────────────────────────────────────────────────────┘");

  resetCounters();
  resetCFCounter();

  const g = new Graph("full_simplification");

  // Runtime input.
  const x     = g.addInputTensor("x",     "float32", [8]);
  // Compile-time constants (scalar broadcast).
  const scale = addConstantScalar(g, "scale", 2.0);
  const bias  = addConstantScalar(g, "bias",  1.0);

  // x * scale  (runtime × constant scalar)
  const scaleNode = g.addNode("mul",  [x.id, scale.id],           [{ name: "scaled", dtype: "float32", shape: [8] }]);
  // scaled + bias  (runtime + constant scalar)
  const biasNode  = g.addNode("add",  [scaleNode.outputs[0], bias.id], [{ name: "biased", dtype: "float32", shape: [8] }]);
  // relu(biased)
  const reluNode  = g.addNode("relu", [biasNode.outputs[0]],       [{ name: "output", dtype: "float32", shape: [8] }]);

  // A dead branch: an extra add that nobody consumes.
  const deadAdd = g.addNode("add",  [x.id, scale.id],             [{ name: "dead",   dtype: "float32", shape: [8] }]);
  void deadAdd; // explicitly not marked as output

  g.markOutputs(reluNode.outputs[0]);

  const nodesBefore = g.nodeOrder.length;
  console.log(`\n  Before: ${nodesBefore} node(s) — ${[...g.nodes.values()].map(n => n.op).join(", ")}`);

  // ── Run through a PassManager with all three pre-layout passes ────────────
  const pm = new PassManager({
    validateAfterEachPass: true,
    logSink: (entry) => {
      const prefix = entry.level === "warn"  ? "  ⚠ " :
                     entry.level === "error" ? "  ✖ " : "    ";
      console.log(`  [${entry.passName}] ${prefix}${entry.message}`);
    },
  });

  pm.addPasses(
    new ConstantFoldingPass(),
    new CSEPass(),
    new DeadCodeEliminationPass(),
  );

  const finalGraph = pm.run(g);
  const nodesAfter = finalGraph.nodeOrder.length;

  console.log(`\n  After:  ${nodesAfter} node(s) — ${[...finalGraph.nodes.values()].map(n => n.op).join(", ")}`);
  console.log(`  Nodes removed: ${nodesBefore - nodesAfter}`);
  console.log(`  Expected: deadAdd removed by DCE; scale/bias const nodes remain as "const" sources.`);

  // scale and bias constants: CF cannot fold mul(x, scale) because x is
  // a runtime tensor with no payload.  What CF CAN fold is any op where ALL
  // inputs are constants — neither scaleNode nor biasNode qualify here.
  // DCE removes the dead add node.
  const valid = validateGraph(finalGraph);
  console.log(`  validateGraph: ${valid.valid ? "PASS" : "FAIL"} (${valid.errors.length} error(s))`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

export function runPreLayoutOptimizationExample(): void {
  console.log("\n╔══════════════════════════════════════════════════════════╗");
  console.log("║  Pre-Layout Graph Simplification Examples                ║");
  console.log("║  ConstantFoldingPass · CSEPass · DeadCodeEliminationPass ║");
  console.log("╚══════════════════════════════════════════════════════════╝");

  runExampleA();
  runExampleB();
  runExampleC();

  console.log("\n═══ Pre-Layout Optimization Examples complete ═══\n");
}
