/**
 * Open string type for tensor memory layouts.
 * Use the Layouts constants for well-known formats; extend with plain strings
 * for backend-specific or custom layouts.
 */
export type LayoutFormat = string;
/**
 * Well-known layout constants.
 * ANY and UNKNOWN have special semantics:
 *   ANY     — "this op accepts any layout" (used in op contracts)
 *   UNKNOWN — "layout not yet determined" (initial state of analysis)
 */
export declare const Layouts: {
    readonly NCHW: "NCHW";
    readonly NHWC: "NHWC";
    readonly NCW: "NCW";
    readonly NWC: "NWC";
    readonly NC: "NC";
    readonly N: "N";
    readonly SCALAR: "SCALAR";
    readonly ANY: "ANY";
    readonly UNKNOWN: "UNKNOWN";
};
/**
 * Axis permutation descriptor.
 * perm[i] = the source axis for destination axis i.
 * Example: NCHW → NHWC is perm = [0, 2, 3, 1].
 */
export interface Permutation {
    readonly perm: readonly number[];
}
/** True iff p is a valid permutation of {0, …, rank-1}. */
export declare function isValidPermutation(p: Permutation, rank: number): boolean;
/** Compute the inverse permutation: if p maps src→dst, the inverse maps dst→src. */
export declare function invertPermutation(p: Permutation): Permutation;
/**
 * Compose two permutations: (first ∘ second).
 * The resulting permutation is equivalent to applying `first` then `second`.
 * Throws when ranks differ.
 */
export declare function composePermutations(first: Permutation, second: Permutation): Permutation;
/** True iff p is the identity permutation [0, 1, 2, …]. */
export declare function isIdentityPermutation(p: Permutation): boolean;
export type LayoutTransformKind = "permute" | "identity" | "unknown";
/**
 * Descriptor for a layout transformation applied by a node (e.g. a transpose).
 * Stored in node attrs so transforms are transparent to generic graph passes.
 */
export interface LayoutTransform {
    readonly kind: LayoutTransformKind;
    readonly fromLayout: LayoutFormat;
    readonly toLayout: LayoutFormat;
    readonly permutation?: Permutation;
}
/** Build a permutation-based LayoutTransform. */
export declare function makePermutationTransform(fromLayout: LayoutFormat, toLayout: LayoutFormat, perm: readonly number[]): LayoutTransform;
/** Build an identity transform (layout unchanged). */
export declare function makeIdentityTransform(layout: LayoutFormat): LayoutTransform;
/**
 * True iff applying `a` then `b` produces the identity transformation.
 * This is the condition required for a transpose pair to be eliminated.
 */
export declare function areInverseTransforms(a: LayoutTransform, b: LayoutTransform): boolean;
/**
 * Compose two adjacent transforms into one.
 * Returns null when:
 *   • The `toLayout` of `first` does not match `fromLayout` of `second`, or
 *   • The combination produces an "unknown" kind.
 * Returns a simplified identity transform when the pair cancels.
 */
export declare function composeTransforms(first: LayoutTransform, second: LayoutTransform): LayoutTransform | null;
/** NCHW → NHWC: move channel axis from dim-1 to dim-3. */
export declare const PERM_NCHW_TO_NHWC: LayoutTransform;
/** NHWC → NCHW: move channel axis from dim-3 to dim-1. */
export declare const PERM_NHWC_TO_NCHW: LayoutTransform;
/**
 * Extract a LayoutTransform descriptor from a node's attrs map.
 * Expected keys: perm (number[]), fromLayout (string), toLayout (string).
 * Returns null when any key is absent or has the wrong type.
 */
export declare function getTransformFromAttrs(attrs: Record<string, unknown>): LayoutTransform | null;
/** Human-readable description of a LayoutTransform. */
export declare function formatTransform(t: LayoutTransform): string;
