import { LoopModule } from "../ir/loopIR";
import { LoopPass, LoopPassResult } from "./loopPass";
export interface LoopFusionConfig {
    /**
     * Run the fusion scan repeatedly until no more pairs can be fused.
     * Useful when three or more adjacent nests benefit from cascaded fusion.
     * Default: true.
     */
    readonly iterateToFixpoint: boolean;
}
export declare const DEFAULT_LOOP_FUSION_CONFIG: LoopFusionConfig;
export declare class LoopFusionPass implements LoopPass {
    readonly name = "LoopFusionPass";
    private readonly config;
    constructor(config?: Partial<LoopFusionConfig>);
    run(module: LoopModule): LoopPassResult;
    private _fuseFunction;
}
