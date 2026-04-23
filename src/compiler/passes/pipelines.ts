// ─────────────────────────────────────────────────────────────────────────────
//
// Pre-built pipeline factories with the correct default pass ordering.
//
// Default order (must be respected for correctness):
//   1. ConstantFoldingPass     — evaluate foldable ops whose inputs are
//      compile-time constants; replace compute nodes with "const" source nodes.
//   2. CSEPass                 — deduplicate nodes that produce identical values
//      (same op + attrs + inputs); rewire consumers to the canonical producer.
//   3. DeadCodeEliminationPass — remove nodes and tensors not reachable from
//      any graph output (includes formerly-live nodes orphaned by CF / CSE).
//   4. LayoutInsertionPass     — insert transpose nodes wherever a layout-sensitive
//      op receives an input with the wrong layout; the compiler owns this work,
//      not the user model.
//   5. LayoutTransformPass     — cancel redundant transpose pairs and propagate
//      layout information so downstream passes see a clean graph.
//   6. FusionPass              — fuse adjacent elementwise and matmul chains
//      after layout noise has been removed.
//   7. LoopLoweringPass        — terminal pass; translates the optimized Graph
//      IR to an explicit Loop IR without mutating the graph.
//
// Usage:
//   const { pm, loopPass } = createDefaultPipeline();
//   const optimized = pm.run(myGraph);
//   const loopModule = loopPass.getLastModule();
// ─────────────────────────────────────────────────────────────────────────────

import { PassManager, PassManagerOptions }              from "./passManager";
import { ConstantFoldingPass }                            from "./constantFoldingPass";
import { CSEPass }                                        from "./csePass";
import { DeadCodeEliminationPass }                        from "./deadCodeEliminationPass";
import { LayoutInsertionPass }                            from "./layoutInsertionPass";
import { LayoutTransformPass }                            from "./layoutTransformPass";
import { FusionPass }                      from "./fusionPass";
import { LoopLoweringPass }                from "./loopLoweringPass";
import { LayoutRuleRegistry }              from "../patterns/layoutRules";
import { RuleRegistry, DEFAULT_FUSION_RULES } from "../patterns/rules";
import { CostModel }                       from "../optimizer/costModel";
import { DEFAULT_CONTRACT_REGISTRY }       from "../ops/opContracts";
import { LoopPassManager, LoopPassManagerOptions } from "./loopPass";
import { LoopFusionPass, LoopFusionConfig }        from "./loopFusionPass";
import { LoopTilingPass, TilingConfig }            from "./loopTilingPass";
import { LoopModule }                              from "../ir/loopIR";

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
 * Create a PassManager preconfigured with the six default passes in the
 * correct order:
 *   ConstantFoldingPass → CSEPass → DeadCodeEliminationPass
 *   → LayoutTransformPass → FusionPass → LoopLoweringPass
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
    // Pre-layout simplification: constant folding → CSE → DCE.
    // Running these three passes before layout analysis ensures that
    // LayoutInsertionPass and LayoutTransformPass operate on the smallest,
    // cleanest possible graph: no redundant constant computations, no
    // duplicate subexpressions, no dead nodes.
    new ConstantFoldingPass(DEFAULT_CONTRACT_REGISTRY),
    new CSEPass(DEFAULT_CONTRACT_REGISTRY),
    new DeadCodeEliminationPass(),
    // Layout-aware passes: insertion of missing transposes, then rewrite/cancel.
    new LayoutInsertionPass(DEFAULT_CONTRACT_REGISTRY),
    new LayoutTransformPass(new LayoutRuleRegistry(), DEFAULT_CONTRACT_REGISTRY),
    new FusionPass(new RuleRegistry(DEFAULT_FUSION_RULES), new CostModel()),
    // Loop IR lowering (terminal — does not mutate the graph).
    loopPass,
  );

  return { pm, loopPass };
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop IR optimization pipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Options accepted by the loop-level pipeline factories.
 * Both sets are optional; defaults are applied when absent.
 */
export interface LoopPipelineOptions {
  fusion?:  Partial<LoopFusionConfig>;
  tiling?:  Partial<TilingConfig>;
  manager?: Partial<LoopPassManagerOptions>;
}

/**
 * Create a LoopPassManager preconfigured with LoopFusionPass followed by
 * LoopTilingPass — the recommended order (fuse adjacent nests first, then
 * tile the resulting larger nests for cache locality).
 *
 * Usage:
 *   const loopPm = createLoopPipeline();
 *   const optimizedModule = loopPm.run(loopPass.getLastModule()!);
 */
export function createLoopPipeline(options: LoopPipelineOptions = {}): LoopPassManager {
  const pm = new LoopPassManager({ validateAfterEachPass: true, ...options.manager });

  pm.addPasses(
    new LoopFusionPass(options.fusion),
    new LoopTilingPass(options.tiling),
  );

  return pm;
}

// ─────────────────────────────────────────────────────────────────────────────
// Full (graph + loop) pipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Container returned by `createFullPipeline()`.
 *
 * Typical usage:
 *   const { pm, loopPass, loopPm } = createFullPipeline();
 *   pm.run(graph);                            // graph optimisation
 *   const module = loopPm.run(loopPass.getLastModule()!); // loop optimisation
 */
export interface FullPipeline {
  /** Graph-level PassManager (LayoutTransform → Fusion → LoopLowering). */
  readonly pm:       PassManager;
  /** The LoopLoweringPass instance; retrieve the LoopModule via getLastModule(). */
  readonly loopPass: LoopLoweringPass;
  /** Loop-level LoopPassManager (LoopFusion → LoopTiling). */
  readonly loopPm:   LoopPassManager;
}

/**
 * Create a full two-stage pipeline: graph optimisation followed by loop
 * IR optimisation.
 *
 * The graph pipeline is identical to `createDefaultPipeline()`.
 * The loop pipeline is identical to `createLoopPipeline()`.
 *
 * @param graphOptions  Forwarded to `createDefaultPipeline()`.
 * @param loopOptions   Forwarded to `createLoopPipeline()`.
 */
export function createFullPipeline(
  graphOptions: Partial<Pick<PassManagerOptions, "validateAfterEachPass" | "logSink">> = {},
  loopOptions:  LoopPipelineOptions = {},
): FullPipeline {
  const { pm, loopPass } = createDefaultPipeline(graphOptions);
  const loopPm           = createLoopPipeline(loopOptions);
  return { pm, loopPass, loopPm };
}

/** Re-export LoopModule for pipeline consumers that need the type. */
export type { LoopModule };
