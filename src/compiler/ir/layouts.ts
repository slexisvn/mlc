// ─────────────────────────────────────────────────────────────────────────────
// Layout model for the ML compiler IR.
//
// A "layout" describes how a tensor's logical dimensions are arranged in memory
// (e.g. NCHW vs NHWC for image data).  The compiler tracks layout facts to:
//   1. Detect conflicts between ops that require different formats.
//   2. Eliminate redundant transpose pairs (NCHW→NHWC→NCHW cancels to identity).
//   3. Propagate layouts through layout-agnostic ops without inserting transposes.
//
// Design choices
// ──────────────
// • LayoutFormat is an open string type — backends can define their own formats
//   without modifying this file (e.g. "CHWN", "NDHWC", vendor-specific codes).
// • Permutations are represented as plain readonly number arrays so they can be
//   embedded in node attrs and serialised to JSON without extra machinery.
// • All helpers are pure functions; no hidden state.
// ─────────────────────────────────────────────────────────────────────────────

// ─── LayoutFormat ────────────────────────────────────────────────────────────

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
export const Layouts = {
  NCHW:    "NCHW",     // batch × channels × height × width (PyTorch default)
  NHWC:    "NHWC",     // batch × height × width × channels (TensorFlow default)
  NCW:     "NCW",      // batch × channels × length  (1-D conv)
  NWC:     "NWC",      // batch × length × channels
  NC:      "NC",       // batch × features  (matmul / linear layers)
  N:       "N",        // batch-only
  SCALAR:  "SCALAR",   // 0-D tensor
  ANY:     "ANY",      // wildcard: op accepts any layout
  UNKNOWN: "UNKNOWN",  // analysis has not determined the layout yet
} as const satisfies Record<string, string>;

// ─── Permutation ─────────────────────────────────────────────────────────────

/**
 * Axis permutation descriptor.
 * perm[i] = the source axis for destination axis i.
 * Example: NCHW → NHWC is perm = [0, 2, 3, 1].
 */
export interface Permutation {
  readonly perm: readonly number[];
}

/** True iff p is a valid permutation of {0, …, rank-1}. */
export function isValidPermutation(p: Permutation, rank: number): boolean {
  if (p.perm.length !== rank) return false;
  const seen = new Set<number>();
  for (const v of p.perm) {
    if (!Number.isInteger(v) || v < 0 || v >= rank || seen.has(v)) return false;
    seen.add(v);
  }
  return true;
}

/** Compute the inverse permutation: if p maps src→dst, the inverse maps dst→src. */
export function invertPermutation(p: Permutation): Permutation {
  const inv = new Array<number>(p.perm.length);
  for (let i = 0; i < p.perm.length; i++) {
    inv[p.perm[i]] = i;
  }
  return { perm: inv };
}

/**
 * Compose two permutations: (first ∘ second).
 * The resulting permutation is equivalent to applying `first` then `second`.
 * Throws when ranks differ.
 */
export function composePermutations(first: Permutation, second: Permutation): Permutation {
  if (first.perm.length !== second.perm.length) {
    throw new Error(
      `Cannot compose permutations of different ranks: ` +
      `${first.perm.length} vs ${second.perm.length}`,
    );
  }
  return { perm: second.perm.map(i => first.perm[i]) };
}

/** True iff p is the identity permutation [0, 1, 2, …]. */
export function isIdentityPermutation(p: Permutation): boolean {
  return p.perm.every((v, i) => v === i);
}

// ─── LayoutTransform ─────────────────────────────────────────────────────────

export type LayoutTransformKind = "permute" | "identity" | "unknown";

/**
 * Descriptor for a layout transformation applied by a node (e.g. a transpose).
 * Stored in node attrs so transforms are transparent to generic graph passes.
 */
export interface LayoutTransform {
  readonly kind:         LayoutTransformKind;
  readonly fromLayout:   LayoutFormat;
  readonly toLayout:     LayoutFormat;
  readonly permutation?: Permutation;
}

/** Build a permutation-based LayoutTransform. */
export function makePermutationTransform(
  fromLayout: LayoutFormat,
  toLayout:   LayoutFormat,
  perm:       readonly number[],
): LayoutTransform {
  return { kind: "permute", fromLayout, toLayout, permutation: { perm: [...perm] } };
}

/** Build an identity transform (layout unchanged). */
export function makeIdentityTransform(layout: LayoutFormat): LayoutTransform {
  return { kind: "identity", fromLayout: layout, toLayout: layout };
}

/**
 * True iff applying `a` then `b` produces the identity transformation.
 * This is the condition required for a transpose pair to be eliminated.
 */
export function areInverseTransforms(a: LayoutTransform, b: LayoutTransform): boolean {
  if (a.toLayout   !== b.fromLayout) return false;
  if (a.fromLayout !== b.toLayout)   return false;
  if (a.kind === "identity" && b.kind === "identity") return true;
  if (a.kind === "permute"  && b.kind === "permute"  &&
      a.permutation          && b.permutation) {
    const composed = composePermutations(a.permutation, b.permutation);
    return isIdentityPermutation(composed);
  }
  return false;
}

/**
 * Compose two adjacent transforms into one.
 * Returns null when:
 *   • The `toLayout` of `first` does not match `fromLayout` of `second`, or
 *   • The combination produces an "unknown" kind.
 * Returns a simplified identity transform when the pair cancels.
 */
export function composeTransforms(
  first:  LayoutTransform,
  second: LayoutTransform,
): LayoutTransform | null {
  if (first.toLayout !== second.fromLayout) return null;
  if (first.kind  === "identity") return second;
  if (second.kind === "identity") return first;
  if (first.kind  === "permute" && second.kind === "permute" &&
      first.permutation          && second.permutation) {
    const composed = composePermutations(first.permutation, second.permutation);
    if (isIdentityPermutation(composed)) {
      return makeIdentityTransform(first.fromLayout);
    }
    return {
      kind:        "permute",
      fromLayout:  first.fromLayout,
      toLayout:    second.toLayout,
      permutation: composed,
    };
  }
  return null;
}

// ─── Well-known transforms ────────────────────────────────────────────────────

/** NCHW → NHWC: move channel axis from dim-1 to dim-3. */
export const PERM_NCHW_TO_NHWC = makePermutationTransform(
  Layouts.NCHW, Layouts.NHWC, [0, 2, 3, 1],
);

/** NHWC → NCHW: move channel axis from dim-3 to dim-1. */
export const PERM_NHWC_TO_NCHW = makePermutationTransform(
  Layouts.NHWC, Layouts.NCHW, [0, 3, 1, 2],
);

// ─── Attrs helpers ────────────────────────────────────────────────────────────

/**
 * Extract a LayoutTransform descriptor from a node's attrs map.
 * Expected keys: perm (number[]), fromLayout (string), toLayout (string).
 * Returns null when any key is absent or has the wrong type.
 */
export function getTransformFromAttrs(attrs: Record<string, unknown>): LayoutTransform | null {
  const perm       = attrs["perm"];
  const fromLayout = attrs["fromLayout"];
  const toLayout   = attrs["toLayout"];
  if (!Array.isArray(perm))           return null;
  if (typeof fromLayout !== "string") return null;
  if (typeof toLayout   !== "string") return null;
  return makePermutationTransform(fromLayout, toLayout, perm as number[]);
}

/** Human-readable description of a LayoutTransform. */
export function formatTransform(t: LayoutTransform): string {
  if (t.kind === "identity") return `identity(${t.fromLayout})`;
  if (t.kind === "permute" && t.permutation) {
    return `permute(${t.fromLayout}→${t.toLayout}, [${t.permutation.perm.join(",")}])`;
  }
  return `unknown(${t.fromLayout}→${t.toLayout})`;
}
