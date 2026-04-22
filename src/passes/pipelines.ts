// ─────────────────────────────────────────────────────────────────────────────
// passes/pipelines.ts
//
// Pre-built pipeline factories with the correct default pass ordering.
//
// Default order (must be respected for correctness):
//   1. LayoutTransformPass — cancel redundant transpose pairs and propagate
//      layout information so downstream passes see a clean graph.
//   2. FusionPass          — fuse adjacent elementwise and matmul chains after
//      layout noise has been removed.
//   3. LoopLoweringPass    — terminal pass; translates the optimized Graph IR
//      to an explicit Loop IR without mutating the graph.
//
// Usage:
//   const { pm, loopPass } = createDefaultPipeline();
//   const optimized = pm.run(myGraph);
//   const loopModule = loopPass.getLastModule();
// ─────────────────────────────────────────────────────────────────────────────

import { PassManager, PassManagerOptions } from "./passManager";
import { LayoutTransformPass }             from "./layoutTransformPass";
import { FusionPass }                      from "./fusionPass";
import { LoopLoweringPass }                from "./loopLoweringPass";
import { LayoutRuleRegistry }              from "../patterns/layoutRules";
import { RuleRegistry, DEFAULT_FUSION_RULES } from "../patterns/rules";
import { CostModel }                       from "../optimizer/costModel";
import { DEFAULT_CONTRACT_REGISTRY }       from "../ops/opContracts";

// ─────────────────────────────────────────────────────────────────────────────

export interface DefaultPipeline {
  /** The configured PassManager.  Call `pm.run(graph)` to execute all passes. */
  readonly pm: PassManager;
  /**
   * The LoopLoweringPass instance in the pipeline.
   * After `pm.run()` completes, call `loopPass.getLastModule()` to retrieve
   * the produced LoopModule.
   */
  readonly loopPass: LoopLoweringPass;
}

/**
 * Create a PassManager preconfigured with the three default passes in the
 * correct order: LayoutTransformPass → FusionPass → LoopLoweringPass.
 *
 * All passes are instantiated with their default registries and cost models.
 * To customize individual passes, build the pipeline manually with
 * `PassManager.addPasses()`.
 *
 * @param options  Forwarded to PassManager.  `validateAfterEachPass` defaults
 *                 to `true` as recommended for development.
 */
export function createDefaultPipeline(
  options: Partial<Pick<PassManagerOptions, "validateAfterEachPass" | "logSink">> = {},
): DefaultPipeline {
  const loopPass = new LoopLoweringPass();

  const pm = new PassManager({ validateAfterEachPass: true, ...options });

  pm.addPasses(
    new LayoutTransformPass(new LayoutRuleRegistry(), DEFAULT_CONTRACT_REGISTRY),
    new FusionPass(new RuleRegistry(DEFAULT_FUSION_RULES), new CostModel()),
    loopPass,
  );

  return { pm, loopPass };
}
