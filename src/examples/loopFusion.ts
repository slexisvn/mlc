// ─────────────────────────────────────────────────────────────────────────────
// examples/loopFusion.ts
//
// Demo: loop-level fusion of two adjacent elementwise nests.
//
// Graph (add followed by relu — deliberately NOT fused at the graph level so
// the two nests arrive at the loop IR as separate perfect nests):
//   a[N], b[N]  →  add  →  tmp[N]  →  relu  →  out[N]
//
// After LoopLoweringPass the LoopFunction body is:
//   for i0 in [0, N):
//     tmp[i0] = a[i0] + b[i0]        ← nest 1
//   for i0 in [0, N):
//     out[i0] = max(0.0, tmp[i0])    ← nest 2
//
// After LoopFusionPass:
//   for i0 in [0, N):
//     tmp[i0] = a[i0] + b[i0]
//     out[i0] = max(0.0, tmp[i0])    ← fused into a single nest
// ─────────────────────────────────────────────────────────────────────────────

import { resetCounters, Graph }  from "../ir/graph";
import { LoopLoweringPass }      from "../passes/loopLoweringPass";
import { PassManager }           from "../passes/passManager";
import { LoopPassManager }       from "../passes/loopPass";
import { LoopFusionPass }        from "../passes/loopFusionPass";
import { printLoopModule }       from "../debug/loopPrinter";
import { printGraph }            from "../debug/printer";

export function runLoopFusionExample(): void {
  resetCounters();

  // ── Build graph: add → relu (two separate ops, NOT graph-fused) ───────────
  const g = new Graph();
  const N = 256;

  const a   = g.addInputTensor("a",   "float32", [N]);
  const b   = g.addInputTensor("b",   "float32", [N]);

  // Using "add" and "relu" as separate ops so the graph-level FusionPass does
  // NOT fire here — we only run LoopLoweringPass below.
  const add  = g.addNode("add",  [a.id, b.id],          [{ name: "tmp", dtype: "float32", shape: [N] }]);
  const relu = g.addNode("relu", [add.outputs[0]],       [{ name: "out", dtype: "float32", shape: [N] }]);
  g.markOutputs(relu.outputs[0]);

  printGraph(g, "Loop Fusion Example — input graph (add + relu, no graph fusion)");

  // ── Step 1: lower to Loop IR (no graph-level fusion pass) ─────────────────
  const lowerPass = new LoopLoweringPass();
  const pm        = new PassManager({ validateAfterEachPass: false });
  pm.addPass(lowerPass);
  pm.run(g);

  const beforeModule = lowerPass.getLastModule()!;
  printLoopModule(beforeModule, "Loop IR — BEFORE loop fusion (two adjacent perfect nests)");

  // ── Step 2: apply LoopFusionPass ──────────────────────────────────────────
  const loopPm = new LoopPassManager({ validateAfterEachPass: true });
  loopPm.addPass(new LoopFusionPass());

  const afterModule = loopPm.run(beforeModule);
  printLoopModule(afterModule, "Loop IR — AFTER loop fusion (single fused nest)");

  // Verify: fused function should have one top-level ForLoop.
  const fn = afterModule.functions[0];
  const topLoops = fn.body.filter(s => s.kind === "ForLoop").length;
  const merged   = topLoops === 1;
  console.log(
    `  ✓ Fusion result: ${topLoops} top-level loop(s) — ` +
    (merged ? "successfully merged into one nest." : "WARNING: expected 1 after fusion."),
  );
}
