// ─────────────────────────────────────────────────────────────────────────────
// examples/layoutCancellation.ts
//
// Demonstrates direct transpose–transpose cancellation.
//
// Graph (before):
//   x[NCHW] → transpose(NCHW→NHWC) → x_nhwc
//           → transpose(NHWC→NCHW) → x_back [output]
//
// After LayoutTransformPass:
//   x[NCHW] → x_back (=x, both transposes eliminated)
// ─────────────────────────────────────────────────────────────────────────────

import { resetCounters, Graph } from "../ir/graph";
import { PassManager } from "../passes/passManager";
import { LayoutTransformPass } from "../passes/layoutTransformPass";
import { LayoutRuleRegistry } from "../patterns/layoutRules";
import { DEFAULT_CONTRACT_REGISTRY } from "../ops/opContracts";
import { analyzeLayouts } from "../analysis/layoutAnalysis";
import { printGraph, printDiff, printLayoutAnalysis } from "../debug/printer";

export function runLayoutCancellationExample(): void {
  resetCounters();

  // ── Build graph ────────────────────────────────────────────────────────────
  const g = new Graph();

  const x     = g.addInputTensor("x_NCHW", "float32", [1, 3, 224, 224]);

  // NCHW → NHWC  (perm = [0, 2, 3, 1])
  const tp1   = g.addNode(
    "transpose",
    [x.id],
    [{ name: "x_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }],
    { perm: [0, 2, 3, 1], fromLayout: "NCHW", toLayout: "NHWC" },
  );

  // NHWC → NCHW  (perm = [0, 3, 1, 2]) — inverse of tp1
  const tp2   = g.addNode(
    "transpose",
    [tp1.outputs[0]],
    [{ name: "x_back_NCHW", dtype: "float32", shape: [1, 3, 224, 224] }],
    { perm: [0, 3, 1, 2], fromLayout: "NHWC", toLayout: "NCHW" },
  );

  g.markOutputs(tp2.outputs[0]);

  // ── Run layout analysis first (informational) ─────────────────────────────
  const layoutFacts = analyzeLayouts(g, DEFAULT_CONTRACT_REGISTRY);
  printLayoutAnalysis(layoutFacts, "Layout Analysis — before cancellation");

  // ── Run LayoutTransformPass ───────────────────────────────────────────────
  const before = g.clone();
  const pm     = new PassManager({ validateAfterEachPass: true });
  pm.addPass(new LayoutTransformPass(new LayoutRuleRegistry(), DEFAULT_CONTRACT_REGISTRY));

  const after  = pm.run(g);

  printGraph(before, "Layout Cancellation — before");
  printGraph(after,  "Layout Cancellation — after");
  printDiff(before, after, "LayoutTransformPass (cancellation)");
}
