import { FusionRule } from "../ir/types";
/**
 * Built-in fusion rules covering the most common DNN patterns.
 * Ordered from longest to shortest within semantic groups so the registry
 * already provides a reasonable greedy-match priority baseline.
 */
export declare const DEFAULT_FUSION_RULES: readonly FusionRule[];
/**
 * Mutable registry of fusion rules.
 *
 * Pass an instance to RuleRegistry to the FusionPass so user code can extend
 * the rule set without touching any pass internals.
 */
export declare class RuleRegistry {
    private readonly _rules;
    constructor(initialRules?: readonly FusionRule[]);
    /** Add a new fusion rule at the end of the registry. */
    addRule(rule: FusionRule): this;
    /** Read-only view of all registered rules. */
    getRules(): readonly FusionRule[];
    /** Number of rules currently registered. */
    get size(): number;
}
