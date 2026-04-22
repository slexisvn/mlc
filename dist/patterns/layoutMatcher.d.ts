import { Graph } from "../ir/graph";
import { LayoutTransform } from "../ir/layouts";
import { OpContractRegistry } from "../ops/opContracts";
import { LayoutRewriteRule } from "./layoutRules";
export interface MatchedLayoutChain {
    readonly rule: LayoutRewriteRule;
    /** Node ids in chain order (first → last). */
    readonly nodeIds: string[];
    /**
     * Layout transforms collected from transforming ops in the chain.
     * For a cancellation rule the first and last transforms should be inverses.
     */
    readonly transforms: LayoutTransform[];
}
/**
 * Find all non-overlapping layout chains in the graph that match a registered
 * LayoutRewriteRule.
 *
 * @param graph     Graph to inspect (not mutated).
 * @param rules     Ordered set of layout rules (from LayoutRuleRegistry.getRules()).
 * @param registry  Op contract registry; defaults to DEFAULT_CONTRACT_REGISTRY.
 */
export declare function matchLayoutChains(graph: Graph, rules: readonly LayoutRewriteRule[], registry?: OpContractRegistry): MatchedLayoutChain[];
