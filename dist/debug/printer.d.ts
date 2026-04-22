import { Graph } from "../ir/graph";
import { LayoutAnalysisResult } from "../analysis/layoutAnalysis";
import { FusionAnalysisResult } from "../analysis/fusionAnalysis";
/**
 * Print a complete human-readable representation of the graph.
 *
 * @param graph  The graph to print.
 * @param title  Optional header shown above the graph dump.
 */
export declare function printGraph(graph: Graph, title?: string): void;
/**
 * Print a topologically-ordered execution plan for the graph.
 * Each step shows the node id, op name, inputs, and outputs.
 */
export declare function printExecutionPlan(graph: Graph, title?: string): void;
/**
 * Print a concise before/after summary showing what the optimiser changed.
 */
export declare function printDiff(before: Graph, after: Graph, passName: string): void;
/**
 * Print a summary of layout facts and any detected conflicts / elimination
 * candidates produced by analyzeLayouts().
 */
export declare function printLayoutAnalysis(result: LayoutAnalysisResult, title?: string): void;
/**
 * Print a summary of fusion analysis results including approved candidates
 * and per-node rejection records.
 */
export declare function printFusionAnalysis(result: FusionAnalysisResult, title?: string): void;
