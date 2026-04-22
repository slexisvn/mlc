import { Graph } from "../ir/graph";
import { FusionRule, ChainCandidate } from "../ir/types";
import { OpContractRegistry } from "../ops/opContracts";
export type RejectionReason = "NoMatchingRule" | "MultiOutputIntermediateNode" | "IntermediateTensorIsGraphOutput" | "BranchingIntermediateTensor" | "NodeAlreadyInChain" | "OpContractForbidsFusion" | "PatternOpMismatch";
export interface RejectionRecord {
    readonly nodeId: string;
    readonly op: string;
    readonly reason: RejectionReason;
    readonly detail: string;
    readonly attemptedRule?: FusionRule;
}
/** A confirmed fusion candidate — superset of ChainCandidate with a mutable nodeIds array. */
export interface CandidateChain extends ChainCandidate {
    readonly rule: FusionRule;
    readonly nodeIds: string[];
}
export interface FusionAnalysisResult {
    readonly candidates: CandidateChain[];
    readonly rejections: RejectionRecord[];
    readonly stats: {
        readonly totalNodes: number;
        readonly fusibleCandidates: number;
        readonly rejectedNodes: number;
    };
}
/**
 * Analyse the graph for fusion opportunities and return both the confirmed
 * candidates (from the matcher) and per-node rejection records for every node
 * that was NOT included in a chain.
 *
 * @param graph     The graph to analyse (not mutated).
 * @param rules     Fusion rules to evaluate (e.g. RuleRegistry.getRules()).
 * @param registry  Op contract registry; defaults to DEFAULT_CONTRACT_REGISTRY.
 */
export declare function analyzeFusion(graph: Graph, rules: readonly FusionRule[], registry?: OpContractRegistry): FusionAnalysisResult;
