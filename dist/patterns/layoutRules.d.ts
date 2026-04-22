export type LayoutRuleKind = "cancellation" | "propagation" | "normalization";
export interface LayoutRewriteRule {
    readonly name: string;
    readonly kind: LayoutRuleKind;
    /** Ordered op names forming the pattern, e.g. ["transpose", "add", "transpose"]. */
    readonly pattern: string[];
    readonly description: string;
    /** Higher value = tried before lower-priority rules. Default: 50. */
    readonly priority: number;
}
export declare const DEFAULT_LAYOUT_RULES: readonly LayoutRewriteRule[];
export declare class LayoutRuleRegistry {
    private readonly _rules;
    constructor(initial?: readonly LayoutRewriteRule[]);
    /** Add a rule and re-sort by priority. Returns `this` for chaining. */
    addRule(rule: LayoutRewriteRule): this;
    getRules(): readonly LayoutRewriteRule[];
    get size(): number;
}
