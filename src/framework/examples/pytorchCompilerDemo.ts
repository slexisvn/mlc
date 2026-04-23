// ─────────────────────────────────────────────────────────────────────────────
// framework/examples/pytorchCompilerDemo.ts
//
// Comprehensive end-to-end demo — exercises all compiler passes on both
// the forward and backward graphs of a model defined via the PyTorch-like API.
//
// Pass coverage plan
// ──────────────────
// ConstantFoldingPass  — ctx.const() tensors flow through add/relu; the
//                        compiler folds them to compile-time values.
// CSEPass              — the model computes x·W twice (shared linear transform);
//                        the second copy is eliminated.
// DeadCodeEliminationPass — a branch is computed but never marked as output;
//                           DCE prunes it (also picks up CF leftovers).
// LayoutInsertionPass  — img is named "img_NHWC" so layout analysis labels it
//                        NHWC.  pool2d requires NCHW.  The compiler detects this
//                        conflict and inserts a NHWC→NCHW transpose automatically.
//                        The user model never calls transposeLayout().
// LayoutTransformPass  — cancels any redundant transpose pairs created during
//                        layout insertion or surviving from prior passes.
// FusionPass           — matmul→add pattern fuses to linear.
// LoopLoweringPass     — always runs; translates surviving nodes to loop IR
//                        (ops without a lowering rule — e.g. transpose, pool2d
//                         — are skipped with a diagnostic warning).
//
// Both the forward graph and the autodiff backward graph are independently run
// through the full pipeline.
// ─────────────────────────────────────────────────────────────────────────────

import * as nn                                              from "../nn";
import { ExportSession }                                    from "../export/session";
import { SymbolicTensor as Tensor }                        from "../tensor/tensor";
import { importGraphIR }                                    from "../export/importGraphIR";
import { IRPackage, GraphIR }                              from "../ir/schema";
import { TensorId }                                        from "../ir/ids";
import { buildBackwardGraph, DEFAULT_GRAD_BUILDERS }        from "../autodiff";
import { defaultOpRegistry }                               from "../core/opRegistry";
import { validateIRPackage }                                from "../ir/validator";
import { validateGraph }                                    from "../../compiler/ir/validate";
import { createDefaultPipeline }                           from "../../compiler/passes/pipelines";
import { printGraph, printDiff }                           from "../../compiler/debug/printer";
import { printLoopModule }                                 from "../../compiler/debug/loopPrinter";

// Register grad builders so autodiff can differentiate all ops used below.
// (The default registry starts empty for grad builders; we extend it here.)
for (const [op, fn] of Object.entries(DEFAULT_GRAD_BUILDERS)) {
  defaultOpRegistry.register({
    ...defaultOpRegistry.get(op),
    gradBuilder: fn,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase A — Model definition
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A model that deliberately exercises every compiler pass trigger pattern.
 *
 * Inputs:
 *   x    [32, 128]    — main activation input
 *
 * Constants (injected via ctx.const):
 *   cA, cB — scalar constants; add(cA, cB).relu() is folded at compile time.
 *
 * The layout insertion trigger (pool2d on img_NHWC) is wired at session.build
 * level so that `pooled` is a graph output and DCE keeps it live.
 */
class AllPassModel extends nn.Module {
  // Fusion trigger: matmul→add pattern fuses to linear
  private readonly fc1 = this.register("fc1", new nn.Linear(128, 64));
  // Second linear for CSE demo (same input weight produces same matmul)
  private readonly fc2 = this.register("fc2", new nn.Linear(64, 32));

  /**
   * @param x   Main input [32, 128]
   * @param cA  Compile-time scalar constant
   * @param cB  Compile-time scalar constant
   */
  forward(x: Tensor, cA: Tensor, cB: Tensor): Tensor {
    // ── Fusion trigger: matmul → add (bias) → linear ──────────────────────
    const h1 = this.fc1.forward(x);          // matmul + add_bias → fuses to linear

    // ── CSE trigger: compute fc1 output again with same inputs ────────────
    // fc1.forward(x) is computed twice → second copy is eliminated by CSE.
    const h1_dup = this.fc1.forward(x);      // duplicate — CSE will remove
    const h2 = this.fc2.forward(h1_dup);     // uses the dup (gets rewired to h1)

    // ── Constant folding + DCE trigger ────────────────────────────────────
    // Both cA and cB carry constantPayload → CF folds the chain at compile time.
    // The result is not included in the return value, so DCE prunes the const
    // node after CF runs.
    const cfResult = cA.add(cB).relu();      // folded then DCE-pruned
    void cfResult;

    return h2;
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

  const fwdSession = new ExportSession({ id: "all_pass_model" });
  const model      = new AllPassModel();

  let fwdGraphIR!: GraphIR;
  let paramIds:    TensorId[] = [];
  let lossId!:     TensorId;

  fwdSession.build(ctx => {
    const x   = ctx.input("x",        "float32", [32, 128]);
    // img_NHWC: name substring "NHWC" seeds layout analysis to NHWC.
    const img = ctx.input("img_NHWC", "float32", [1, 8, 8, 4]);

    // Compile-time constants — bridge will attach constantPayload.
    const cA = ctx.const("cA", "float32", [1], [2.0]);
    const cB = ctx.const("cB", "float32", [1], [3.0]);

    // Layout insertion trigger: pool2d requires NCHW; img has NHWC layout.
    // The user model never calls transposeLayout() — LayoutInsertionPass
    // detects the conflict and inserts NHWC→NCHW automatically.
    // Marking pooled as an output keeps the node live past DCE.
    const pooled = img.pool2d(2, 2);
    ctx.markOutput(pooled);   // first output — DCE-live

    const y = model.forward(x, cA, cB);
    ctx.markOutput(y);        // last output = lossId for autodiff
  });

  const fwdPkg = fwdSession.export("forward");
  fwdGraphIR   = fwdPkg.graphs[0];

  // Collect param ids for autodiff — exclude compile-time constants since
  // they're not on the param → output path that autodiff differentiates.
  const constIds = new Set(
    (fwdPkg.parameters ?? [])
      .filter(p => p.isConst)
      .map(p => p.tensorId),
  );
  paramIds = fwdGraphIR.inputIds.filter(
    tid => fwdGraphIR.tensors[tid].isParam && !constIds.has(tid),
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
    defaultOpRegistry,
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
