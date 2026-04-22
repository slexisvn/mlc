// ─────────────────────────────────────────────────────────────────────────────
// examples/layoutMismatch.ts
//
// Demonstrates layout conflict detection.
//
// Graph:
//   x[NHWC] → conv_strict (requires NCHW only) → y [output]
//
// The LayoutTransformPass does NOT auto-resolve conflicts — it detects and
// reports them.  The graph is passed through unchanged, and the PassManager
// emits a warning so the developer knows a manual fix is needed.
// ─────────────────────────────────────────────────────────────────────────────

import { resetCounters, Graph } from "../ir/graph";
import { PassManager } from "../passes/passManager";
import { LayoutTransformPass } from "../passes/layoutTransformPass";
import { LayoutRuleRegistry } from "../patterns/layoutRules";
import { OpContractRegistry, DEFAULT_OP_CONTRACTS } from "../ops/opContracts";
import { analyzeLayouts } from "../analysis/layoutAnalysis";
import { printGraph, printLayoutAnalysis } from "../debug/printer";

export function runLayoutMismatchExample(): void {
  resetCounters();

  // ── Custom registry: a conv that only accepts NCHW ────────────────────────
  const strictRegistry = new OpContractRegistry(DEFAULT_OP_CONTRACTS);
  strictRegistry.register({
    op:                   "conv_strict",
    fusibilityClass:      "fusible",
    layoutBehavior:       "sensitive",
    requiredInputLayouts: ["NCHW"],   // no NHWC allowed
    description:          "Conv that strictly requires NCHW input.",
  });

  // ── Build graph with an NHWC input fed to a strict-NCHW conv ──────────────
  const g = new Graph();

  const x    = g.addInputTensor("x_NHWC", "float32", [1, 224, 224, 3]);
  const conv = g.addNode(
    "conv_strict",
    [x.id],
    [{ name: "y", dtype: "float32", shape: [1, 64, 224, 224] }],
  );
  g.markOutputs(conv.outputs[0]);

  // ── Run layout analysis to surface the conflict ───────────────────────────
  const facts = analyzeLayouts(g, strictRegistry);
  printLayoutAnalysis(facts, "Layout Mismatch — conflict report");
  printGraph(g, "Layout Mismatch — graph (unchanged)");

  // ── Run LayoutTransformPass — it warns but does not crash ─────────────────
  const pm = new PassManager({ validateAfterEachPass: true });
  pm.addPass(new LayoutTransformPass(new LayoutRuleRegistry(), strictRegistry));
  pm.run(g);   // graph unchanged; conflicts logged

  console.log("  Note: the graph was not rewritten — a manual transpose insertion");
  console.log("        is required to satisfy the layout contract of conv_strict.\n");
}
