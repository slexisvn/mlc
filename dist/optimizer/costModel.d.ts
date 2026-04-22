import { Graph } from "../ir/graph";
import { ChainCandidate } from "../ir/types";
import { MatchedChain } from "../patterns/matcher";
export interface CostModelConfig {
    /**
     * Maximum allowed chain length before the cost model rejects fusion.
     * Very long chains can produce kernels that are hard to lower to hardware.
     * Default: 8 (covers all DEFAULT_FUSION_RULES comfortably).
     */
    maxChainLength: number;
}
export declare const DEFAULT_COST_MODEL_CONFIG: CostModelConfig;
/**
 * Structured profitability evaluation returned by CostModel.evaluate().
 * The `reason` string is human-readable and used in diagnostic logs.
 */
export interface CostModelEvaluation {
    readonly shouldFuse: boolean;
    readonly nodeCountReduction: number;
    readonly reason: string;
}
export declare class CostModel {
    private readonly config;
    constructor(config?: Partial<CostModelConfig>);
    /**
     * Decide whether fusing the given chain is profitable and safe.
     *
     * @param chain  A matched chain candidate (already validated by the matcher).
     * @param graph  The current graph (used for double-checking consumer invariants).
     * @returns true if the fusion should proceed.
     */
    shouldFuse(chain: MatchedChain, graph: Graph): boolean;
    /**
     * Return a structured evaluation with both the decision and the reason.
     * Accepts any ChainCandidate (MatchedChain is a subtype) for composability
     * with the analysis layer.
     */
    evaluate(chain: ChainCandidate, graph: Graph): CostModelEvaluation;
}
