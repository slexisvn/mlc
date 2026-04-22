// ─────────────────────────────────────────────────────────────────────────────
// examples/frameworkDemo.ts
//
// End-to-end demonstration of the mini deep learning framework.
//
// This example:
//   1. Builds a 2-layer MLP using the Module API.
//   2. Also builds the same network using the functional ops API directly.
//   3. Serialises the forward graph to JSON and round-trips through
//      deserialisation + IR validation.
//   4. Imports the GraphIR through the bridge into a compiler Graph.
//   5. Runs the default compiler pipeline (CF → CSE → DCE → Layout →
//      Fusion → LoopLowering) on the imported graph.
//   6. Prints a summary of each stage.
// ─────────────────────────────────────────────────────────────────────────────

import { GraphBuilder }                             from "../framework/graphBuilder";
import { ParameterStore }                           from "../framework/parameter";
import { MLP, Linear }                              from "../framework/module";
import { matmul, relu, sigmoid, add, sum, reshape } from "../framework/ops";
import { serializeToJSON, deserializeFromJSON }     from "../shared-ir/serializer";
import { validateIRPackage }                        from "../shared-ir/validator";
import { importGraphIR }                            from "../bridge/importGraphIR";
import { createDefaultPipeline }                    from "../passes/pipelines";
import { validateGraph }                            from "../ir/validate";

// ─────────────────────────────────────────────────────────────────────────────

function section(title: string): void {
  console.log(`\n${"─".repeat(60)}`);
  console.log(`  ${title}`);
  console.log("─".repeat(60));
}

// ─── Example A: Module API — MLP ─────────────────────────────────────────────

function exampleMLP(): void {
  section("A: MLP via Module API");

  const gb    = new GraphBuilder({ id: "mlp_fwd" });
  const store = new ParameterStore();

  // Data input: batch of 32 samples with 784 features (e.g. MNIST)
  const x = gb.input("x", "float32", [32, 784]);

  // Build a 3-layer MLP: 784 → 256 → 128 → 10
  const mlp = new MLP(784, [256, 128], 10, gb, store, { activation: "relu" });

  const logits = mlp.forward(x) as ReturnType<typeof mlp.forward>;
  gb.markOutputs(Array.isArray(logits) ? logits[0] : logits);

  const graphIR = gb.build("forward");

  console.log(`\nGraph "${graphIR.id}":`);
  console.log(`  Inputs:  ${graphIR.inputIds.length} tensors`);
  console.log(`  Nodes:   ${graphIR.nodeOrder.length}`);
  console.log(`  Outputs: ${graphIR.outputIds.length} tensors`);
  console.log(`  Params:  ${store.size}`);

  // Serialise
  const paramData = store.toParameterData();
  const json      = serializeToJSON([graphIR], paramData, { pretty: false });
  console.log(`  JSON length: ${json.length} chars`);

  // Validate
  const pkg        = deserializeFromJSON(json);
  const validation = validateIRPackage(pkg);
  console.log(`  IR validation: ${validation.valid ? "PASS" : "FAIL"}`);
  if (!validation.valid) {
    for (const e of validation.errors) {
      console.log(`    ✗ [${e.kind}]: ${e.message}`);
    }
  }

  // Import into compiler
  const { graph: compilerGraph, tensorIdMap } = importGraphIR(pkg);
  const graphValidation = validateGraph(compilerGraph);
  console.log(`  Compiler graph validation: ${graphValidation.valid ? "PASS" : "FAIL"}`);
  console.log(`  Compiler nodes: ${compilerGraph.nodeOrder.length}`);
  console.log(`  Tensor map size: ${tensorIdMap.size}`);

  // Run default pipeline
  const { pm } = createDefaultPipeline();
  const optimised = pm.run(compilerGraph);
  console.log(`  Optimised nodes: ${optimised.nodeOrder.length}`);

  console.log("\n  MLP example: complete");
}

// ─── Example B: Functional API — linear regression ───────────────────────────

function exampleLinearRegression(): void {
  section("B: Linear regression via functional API");

  const gb    = new GraphBuilder({ id: "linreg_fwd" });

  // Inputs
  const x = gb.input("x", "float32", [16, 10]); // batch=16, features=10
  const w = gb.param("w", "float32", [10, 1]);   // weight column
  const b = gb.param("b", "float32", [1]);        // bias scalar

  // y_hat = x @ w + b
  const mm    = matmul(gb, x, w);                 // [16, 1]
  const yHat  = add(gb, mm, b);                   // [16, 1]

  gb.markOutputs(yHat);

  const graphIR = gb.build("forward");

  console.log(`\nGraph "${graphIR.id}":`);
  console.log(`  Nodes:   ${graphIR.nodeOrder.length}`);
  console.log(`  Outputs: ${graphIR.outputIds.length}`);
  console.log(`  Output shape: [${graphIR.tensors[graphIR.outputIds[0]].shape.join(", ")}]`);

  // Bridge + validate
  const { graph: compilerGraph } = importGraphIR({
    irVersion:    "0.1",
    opsetVersion: "mini-ts-0.1",
    graphs:       [graphIR],
  });

  const vr = validateGraph(compilerGraph);
  console.log(`  Compiler graph validation: ${vr.valid ? "PASS" : "FAIL"}`);

  // Run compiler
  const { pm } = createDefaultPipeline();
  const optimised = pm.run(compilerGraph);
  console.log(`  Compiler nodes after optimisation: ${optimised.nodeOrder.length}`);

  console.log("\n  Linear regression example: complete");
}

// ─── Example C: Functional API — shape inference chain ───────────────────────

function exampleShapeInference(): void {
  section("C: Shape inference chain");

  const gb = new GraphBuilder({ id: "shapes" });

  // [4, 3, 8] input
  const x     = gb.input("x",   "float32", [4, 3, 8]);
  // reshape to [4, 24]
  const flat  = reshape(gb, x, { shape: [4, -1] });
  // matmul [4,24] × [24,16] → [4,16]
  const w     = gb.param("w",   "float32", [24, 16]);
  const mm    = matmul(gb, flat, w);
  // relu → sigmoid
  const r     = relu(gb, mm);
  const s     = sigmoid(gb, r);
  // sum over axis 1 → [4]
  const loss  = sum(gb, s, { axes: [1] });

  gb.markOutputs(loss);
  const graphIR = gb.build("forward");

  console.log("\nShape inference trace:");
  for (const nid of graphIR.nodeOrder) {
    const node = graphIR.nodes[nid];
    const outShapes = node.outputs.map(tid =>
      `[${graphIR.tensors[tid].shape.join(", ")}]`,
    ).join(", ");
    console.log(`  ${node.op.padEnd(12)} → ${outShapes}`);
  }

  console.log(`\n  Output tensor shape: [${graphIR.tensors[graphIR.outputIds[0]].shape.join(", ")}]`);
  console.log("  Shape inference example: complete");
}

// ─── Runner ───────────────────────────────────────────────────────────────────

export function runFrameworkDemo(): void {
  console.log("\n══════════════════════════════════════════════════════════════");
  console.log("  Mini Deep Learning Framework — End-to-End Demo");
  console.log("══════════════════════════════════════════════════════════════");

  exampleMLP();
  exampleLinearRegression();
  exampleShapeInference();

  console.log("\n══════════════════════════════════════════════════════════════");
  console.log("  Framework demo complete");
  console.log("══════════════════════════════════════════════════════════════\n");
}
