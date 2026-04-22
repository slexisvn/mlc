import { Graph } from "../ir/graph";
import { FusionRule } from "../ir/types";
export interface MatchedChain {
    /** The rule that produced this match. */
    rule: FusionRule;
    /**
     * Node ids in chain order (first → last), corresponding to rule.pattern elements.
     * Length always equals rule.pattern.length.
     */
    nodeIds: string[];
}
/**
 * Find all non-overlapping fusible chains in the graph using the given rules.
 *
 * Returns an ordered list of MatchedChain objects ready for the FusionPass to
 * process.  No chain shares a node with another (greedy, first-match wins per
 * start node).
 */
export declare function matchChains(graph: Graph, rules: readonly FusionRule[]): MatchedChain[];
