// ─────────────────────────────────────────────────────────────────────────────
// examples/layoutPropagation.ts
//
// Demonstrates transpose sandwich removal around a layout-agnostic add.
//
// Graph (before):
//   x[NCHW] → transpose(NCHW→NHWC) → x_nhwc
//   x_nhwc, bias → add → sum_NHWC
//   sum_NHWC → transpose(NHWC→NCHW) → result [output]
//
// The add op is layout-agnostic, so both transposes are unnecessary.
//
// After LayoutTransformPass:
//   x[NCHW], bias → add → sum_NHWC [output]
//   (the add now operates in NCHW layout; both transposes eliminated)
// ─────────────────────────────────────────────────────────────────────────────

import { resetCounters, Graph } from "../ir/graph";
import { PassManager } from "../passes/passManager";
import { LayoutTransformPass } from "../passes/layoutTransformPass";
import { LayoutRuleRegistry } from "../patterns/layoutRules";
import { DEFAULT_CONTRACT_REGISTRY } from "../ops/opContracts";
import { printGraph, printDiff } from "../debug/printer";

export function runLayoutPropagationExample(): void {
  resetCounters();

  // ── Build graph ────────────────────────────────────────────────────────────
  const g = new Graph();

  const x    = g.addInputTensor("x_NCHW",  "float32", [1, 3, 224, 224]);
  const bias = g.addInputTensor("bias",     "float32", [1, 3, 224, 224]);

  // NCHW → NHWC
  const tp1  = g.addNode(
    "transpose",
    [x.id],
    [{ name: "x_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }],
    { perm: [0, 2, 3, 1], fromLayout: "NCHW", toLayout: "NHWC" },
  );

  // add(x_nhwc, bias) — layout-agnostic
  const add  = g.addNode(
    "add",
    [tp1.outputs[0], bias.id],
    [{ name: "sum_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }],
  );

  // NHWC → NCHW  (inverse of tp1)
  const tp2  = g.addNode(
    "transpose",
    [add.outputs[0]],
    [{ name: "result_NCHW", dtype: "float32", shape: [1, 3, 224, 224] }],
    { perm: [0, 3, 1, 2], fromLayout: "NHWC", toLayout: "NCHW" },
  );

  g.markOutputs(tp2.outputs[0]);

  // ── Run LayoutTransformPass ───────────────────────────────────────────────
  const before = g.clone();
  const pm     = new PassManager({ validateAfterEachPass: true });
  pm.addPass(new LayoutTransformPass(new LayoutRuleRegistry(), DEFAULT_CONTRACT_REGISTRY));

  const after  = pm.run(g);

  printGraph(before, "Layout Propagation — before");
  printGraph(after,  "Layout Propagation — after");
  printDiff(before, after, "LayoutTransformPass (propagation)");
}
