import { LayoutTransform } from "../ir/layouts";
export interface CanonicalizationResult {
    readonly simplified: LayoutTransform[];
    /** How many transforms were eliminated versus the input. */
    readonly eliminated: number;
    readonly description: string;
}
/**
 * Simplify a chain of LayoutTransforms by composing adjacent pairs and
 * removing identities.  Returns a CanonicalizationResult with the simplified
 * chain and a human-readable description of what changed.
 */
export declare function simplifyTransformChain(transforms: readonly LayoutTransform[]): CanonicalizationResult;
/**
 * Quick test: do two adjacent transforms cancel completely to an identity?
 * This is the precondition checked before a cancellation graph rewrite.
 */
export declare function transformsCancel(t1: LayoutTransform, t2: LayoutTransform): boolean;
/**
 * Try to simplify a pair of adjacent transforms into a single transform.
 * Returns null when composition is not possible or produces no simplification.
 */
export declare function simplifyPair(t1: LayoutTransform, t2: LayoutTransform): LayoutTransform | null;
