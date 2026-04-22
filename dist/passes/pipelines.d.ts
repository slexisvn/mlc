import { PassManager, PassManagerOptions } from "./passManager";
import { LoopLoweringPass } from "./loopLoweringPass";
import { LoopPassManager, LoopPassManagerOptions } from "./loopPass";
import { LoopFusionConfig } from "./loopFusionPass";
import { TilingConfig } from "./loopTilingPass";
import { LoopModule } from "../ir/loopIR";
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
export declare function createDefaultPipeline(options?: Partial<Pick<PassManagerOptions, "validateAfterEachPass" | "logSink">>): DefaultPipeline;
/**
 * Options accepted by the loop-level pipeline factories.
 * Both sets are optional; defaults are applied when absent.
 */
export interface LoopPipelineOptions {
    fusion?: Partial<LoopFusionConfig>;
    tiling?: Partial<TilingConfig>;
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
export declare function createLoopPipeline(options?: LoopPipelineOptions): LoopPassManager;
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
    readonly pm: PassManager;
    /** The LoopLoweringPass instance; retrieve the LoopModule via getLastModule(). */
    readonly loopPass: LoopLoweringPass;
    /** Loop-level LoopPassManager (LoopFusion → LoopTiling). */
    readonly loopPm: LoopPassManager;
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
export declare function createFullPipeline(graphOptions?: Partial<Pick<PassManagerOptions, "validateAfterEachPass" | "logSink">>, loopOptions?: LoopPipelineOptions): FullPipeline;
/** Re-export LoopModule for pipeline consumers that need the type. */
export type { LoopModule };
