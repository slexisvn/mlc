// ─────────────────────────────────────────────────────────────────────────────
//
// End-to-end demo: regular deeper MLP with ReLU activations.
//
// Model: MLPClassifier
//   x  →  fc1 (128→256) → ReLU  →  fc2 (256→128) → ReLU  →  fc3 (128→32)
//
// Compiler optimisations demonstrated
// ─────────────────────────────────────
// FusionPass       — matmul+bias+relu fuses to linear_relu (fc1, fc2).
//                    matmul+bias fuses to linear (fc3).
// DCE              — removes unused graph inputs from the optimised backward
//                    graph (forward inputs not consumed by any gradient node).
// LoopLoweringPass — emits explicit loop nests for linear, linear_relu, and
//                    the step op (Heaviside) introduced by the relu gradient.
//
// Backward differentiation
// ─────────────────────────
// ReLU gradient: ∂L/∂x = ∂L/∂out * step(x)   (Heaviside gate).
// Gradients are computed for all six trainable parameters:
//   fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias.
//
// Note: no optimizer or training loop is included — the framework exposes
// forward tracing and backward graph generation only.
// ─────────────────────────────────────────────────────────────────────────────

import * as nn                                              from "../framework/nn";
import { ExportSession }                                    from "../framework/export/session";
import { SymbolicTensor as Tensor }                        from "../framework/tensor/tensor";
import { importGraphIR }                                    from "../framework/export/importGraphIR";
import { IRPackage, GraphIR }                              from "../framework/ir/schema";
import { TensorId }                                        from "../framework/ir/ids";
import { buildBackwardGraph }                               from "../framework/autodiff";
import { validateIRPackage }                                from "../framework/ir/validator";
import { validateGraph }                                    from "../compiler/ir/validate";
import { createDefaultPipeline }                           from "../compiler/passes/pipelines";
import { printGraph, printDiff }                           from "../compiler/debug/printer";
import { printLoopModule }                                 from "../compiler/debug/loopPrinter";

// ─────────────────────────────────────────────────────────────────────────────
// Phase A — Model definition
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Three-layer MLP with ReLU activations.
 *
 *   x [32,128]  →  fc1(128→256)+ReLU  →  fc2(256→128)+ReLU  →  fc3(128→32)
 *
 * The final layer has no activation so logits are unbounded.
 */
class MLPClassifier extends nn.Module {
  private readonly fc1 = this.register("fc1", new nn.Linear(128, 256));
  private readonly fc2 = this.register("fc2", new nn.Linear(256, 128));
  private readonly fc3 = this.register("fc3", new nn.Linear(128, 32));

  forward(x: Tensor): Tensor {
    const h1 = this.fc1.forward(x).relu();   // [32, 256]  → fused to linear_relu
    const h2 = this.fc2.forward(h1).relu();  // [32, 128]  → fused to linear_relu
    return this.fc3.forward(h2);             // [32, 32]   → fused to linear
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo entry point
// ─────────────────────────────────────────────────────────────────────────────

export function runPytorchCompilerDemo(): void {
  const SEP = "═".repeat(66);
  const sec = (title: string) => console.log(`\n${SEP}\n  ${title}\n${SEP}`);

  // ── A: Trace forward graph ─────────────────────────────────────────────
  sec("Phase A — Model definition (PyTorch-like API)");

  const fwdSession = new ExportSession({ id: "mlp_classifier" });
  const model      = new MLPClassifier();

  let fwdGraphIR!: GraphIR;
  let paramIds:    TensorId[] = [];
  let lossId!:     TensorId;

  fwdSession.build(ctx => {
    const x = ctx.input("x", "float32", [32, 128]);
    const y = model.forward(x);
    ctx.markOutput(y);   // [32, 32] — seeded as the loss root for autodiff
  });

  const fwdPkg = fwdSession.export("forward");
  fwdGraphIR   = fwdPkg.graphs[0];

  // All graph inputs that are trainable parameters become autodiff targets.
  paramIds = fwdGraphIR.inputIds.filter(
    tid => fwdGraphIR.tensors[tid].isParam,
  );

  // Loss id: last output of forward graph
  lossId = fwdGraphIR.outputIds[fwdGraphIR.outputIds.length - 1];

  console.log(
    `  ✓ Forward graph built — ` +
    `${Object.keys(fwdGraphIR.nodes).length} nodes, ` +
    `${fwdPkg.parameters?.length ?? 0} parameters`,
  );

  // ── B: Validate forward IR ────────────────────────────────────────────
  sec("Phase B — Forward IR validation");
  const fwdValidation = validateIRPackage(fwdPkg);
  if (!fwdValidation.valid) {
    console.error("  ✗ Forward IRPackage validation failed:");
    for (const e of fwdValidation.errors) console.error(`    [${e.kind}] ${e.message}`);
    return;
  }
  console.log("  ✓ Forward IRPackage is valid");

  // ── C: Build backward graph ───────────────────────────────────────────
  sec("Phase C — Autodiff: build backward graph");

  const { backwardGraph, gradMap } = buildBackwardGraph(
    fwdGraphIR,
    [lossId],
    paramIds,
  );

  console.log(
    `  ✓ Backward graph built — ` +
    `${Object.keys(backwardGraph.nodes).length} nodes, ` +
    `${gradMap.size} parameter gradient(s)`,
  );
  console.log(`  Gradient map (fwd-param-id → bwd-grad-id):`);
  for (const [pid, gid] of gradMap) {
    const pname = fwdGraphIR.tensors[pid]?.name ?? pid;
    console.log(`    ${pname} (${pid}) → grad ${gid}`);
  }

  // Assemble full package with both graphs
  const fullPkg: IRPackage = {
    irVersion:    fwdPkg.irVersion,
    opsetVersion: fwdPkg.opsetVersion,
    graphs:       [fwdPkg.graphs[0], backwardGraph],
    parameters:   fwdPkg.parameters,
  };

  // ── D: Compile FORWARD graph ──────────────────────────────────────────
  sec("Phase D — Compiler optimisation: FORWARD graph");
  _compileAndPrint("forward", fullPkg, sec);

  // ── E: Compile BACKWARD graph ─────────────────────────────────────────
  sec("Phase E — Compiler optimisation: BACKWARD graph");
  _compileAndPrint("backward", fullPkg, sec);

  console.log(`\n  Demo complete.\n`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: import → validate → run all passes → print diff + loop IR
// ─────────────────────────────────────────────────────────────────────────────

function _compileAndPrint(
  kind:    "forward" | "backward",
  pkg:     IRPackage,
  sec:     (t: string) => void,
): void {
  const { graph: inputGraph } = importGraphIR(pkg, { kind });

  const graphValidation = validateGraph(inputGraph);
  if (!graphValidation.valid) {
    console.error(`  ✗ ${kind} compiler graph validation failed:`);
    for (const e of graphValidation.errors) console.error(`    [${e.kind}] ${e.message}`);
    return;
  }
  console.log(`  ✓ ${kind} compiler graph valid — ${inputGraph.nodes.size} nodes`);
  printGraph(inputGraph, `${kind} input graph`);

  const logs: Array<{ pass: string; level: string; message: string }> = [];

  const { pm, loopPass } = createDefaultPipeline({
    validateAfterEachPass: true,
    logSink: entry => {
      logs.push({ pass: entry.passName, level: entry.level, message: entry.message });
      const icon = entry.level === "error" ? "✗" : entry.level === "warn" ? "⚠" : "·";
      console.log(`  [${entry.passName}] ${icon} ${entry.message}`);
    },
  });

  const optimisedGraph = pm.run(inputGraph);

  const uniquePasses = [...new Set(logs.map(l => l.pass))].length;
  console.log(`\n  ✓ ${kind} pipeline complete. Passes run: ${uniquePasses}`);

  printGraph(optimisedGraph, `${kind} optimised graph`);
  printDiff(inputGraph, optimisedGraph, `${kind} full pipeline diff`);

  const loopModule = loopPass.getLastModule();
  if (loopModule) {
    printLoopModule(loopModule, `Loop IR — ${kind}`);
  }
}

runPytorchCompilerDemo();