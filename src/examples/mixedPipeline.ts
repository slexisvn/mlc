// ─────────────────────────────────────────────────────────────────────────────
// examples/mixedPipeline.ts
//
// End-to-end demo: LayoutTransformPass → FusionPass → LoopLoweringPass.
//
// Graph (before):
//   x[NCHW], bias → transpose(NCHW→NHWC)
//                 → add(x_nhwc, bias)
//                 → transpose(NHWC→NCHW)
//                 → relu
//                 → [output]
//
// After LayoutTransformPass:
//   x[NCHW], bias → add → relu → [output]
//   (sandwich transposes cancelled by propagation rule)
//
// After FusionPass:
//   x[NCHW], bias → add_relu → [output]
//   (add→relu fused by the add_relu rule)
//
// After LoopLoweringPass (terminal — graph unchanged):
//   One LoopFunction with a single fused loop nest — no intermediate buffers.
//   out[i0,i1,i2,i3] = max(0.0, (x_NCHW[…] + bias[…]))
// ─────────────────────────────────────────────────────────────────────────────

import { resetCounters, Graph } from "../ir/graph";
import { PassManager } from "../passes/passManager";
import { LayoutTransformPass } from "../passes/layoutTransformPass";
import { LayoutRuleRegistry } from "../patterns/layoutRules";
import { RuleRegistry, DEFAULT_FUSION_RULES } from "../patterns/rules";
import { DEFAULT_CONTRACT_REGISTRY } from "../ops/opContracts";
import { analyzeFusion } from "../analysis/fusionAnalysis";
import { printGraph, printDiff, printFusionAnalysis } from "../debug/printer";
import { printLoopModule } from "../debug/loopPrinter";
import { createDefaultPipeline } from "../passes/pipelines";

export function runMixedPipelineExample(): void {
  resetCounters();

  // ── Build graph ────────────────────────────────────────────────────────────
  const g    = new Graph();

  const x    = g.addInputTensor("x_NCHW",  "float32", [1, 3, 224, 224]);
  const bias = g.addInputTensor("bias",     "float32", [1, 3, 224, 224]);

  const tp1  = g.addNode(
    "transpose",
    [x.id],
    [{ name: "x_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }],
    { perm: [0, 2, 3, 1], fromLayout: "NCHW", toLayout: "NHWC" },
  );
  const add  = g.addNode(
    "add",
    [tp1.outputs[0], bias.id],
    [{ name: "sum_NHWC", dtype: "float32", shape: [1, 224, 224, 3] }],
  );
  const tp2  = g.addNode(
    "transpose",
    [add.outputs[0]],
    [{ name: "sum_NCHW", dtype: "float32", shape: [1, 3, 224, 224] }],
    { perm: [0, 3, 1, 2], fromLayout: "NHWC", toLayout: "NCHW" },
  );
  const relu = g.addNode(
    "relu",
    [tp2.outputs[0]],
    [{ name: "out", dtype: "float32", shape: [1, 3, 224, 224] }],
  );
  g.markOutputs(relu.outputs[0]);

  const original = g.clone();

  // ── Intermediate: layout-clean graph for fusion diagnostics ───────────────
  const afterLayout = (() => {
    const singleStage = new PassManager({ validateAfterEachPass: true });
    singleStage.addPass(
      new LayoutTransformPass(new LayoutRuleRegistry(), DEFAULT_CONTRACT_REGISTRY),
    );
    return singleStage.run(g.clone());
  })();

  const rules          = new RuleRegistry(DEFAULT_FUSION_RULES);
  const fusionAnalysis = analyzeFusion(afterLayout, rules.getRules(), DEFAULT_CONTRACT_REGISTRY);
  printFusionAnalysis(fusionAnalysis, "Fusion Analysis (after layout pass)");

  // ── Full 3-stage pipeline: Layout → Fusion → LoopLowering ────────────────
  const { pm, loopPass } = createDefaultPipeline();
  const final = pm.run(original);

  printGraph(original, "Mixed Pipeline — original");
  printGraph(final,    "Mixed Pipeline — after layout + fusion");
  printDiff(original, final, "LayoutTransformPass + FusionPass + LoopLoweringPass");

  // ── Loop IR — the payoff: one fused nest, no intermediate buffers ─────────
  const loopModule = loopPass.getLastModule();
  if (loopModule) {
    printLoopModule(loopModule, "Loop IR — fused add_relu kernel");
  }
}
