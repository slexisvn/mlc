import { Graph } from "../ir/graph";
import { LayoutFormat, LayoutTransform } from "../ir/layouts";
import { OpContractRegistry } from "../ops/opContracts";
export interface TensorLayoutFact {
    readonly tensorId: string;
    readonly layout: LayoutFormat;
    /** Confidence in the layout determination. */
    readonly confidence: "certain" | "inferred" | "unknown";
    /** How the layout was determined. */
    readonly source: "annotation" | "propagated" | "default";
}
export interface LayoutConflict {
    readonly nodeId: string;
    readonly op: string;
    readonly inputTensorId: string;
    readonly actualLayout: LayoutFormat;
    readonly requiredLayout: string;
    readonly message: string;
}
export interface EliminationCandidate {
    readonly firstNodeId: string;
    readonly secondNodeId: string;
    readonly transform1: LayoutTransform;
    readonly transform2: LayoutTransform;
    readonly reason: string;
}
export interface LayoutAnalysisResult {
    readonly tensorFacts: Map<string, TensorLayoutFact>;
    readonly conflicts: LayoutConflict[];
    readonly eliminationCandidates: EliminationCandidate[];
}
export declare function analyzeLayouts(graph: Graph, registry?: OpContractRegistry): LayoutAnalysisResult;
