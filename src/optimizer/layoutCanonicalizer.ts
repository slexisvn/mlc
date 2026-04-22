// ─────────────────────────────────────────────────────────────────────────────
// optimizer/layoutCanonicalizer.ts
//
// Pure-function utilities for simplifying chains of LayoutTransforms.
//
// The canonicalization algorithm repeatedly scans an ordered list of
// LayoutTransform objects and:
//   1. Composes adjacent transforms where possible (A ∘ B → AB).
//   2. Removes any resulting identity transforms (unless the list would become empty).
//
// The loop continues until no further simplification is possible.
//
// These helpers are used by LayoutTransformPass to verify that a matched chain
// truly cancels before committing to a graph rewrite.
// ─────────────────────────────────────────────────────────────────────────────

import {
  LayoutTransform,
  composeTransforms,
  areInverseTransforms,
  formatTransform,
} from "../ir/layouts";

// ─── Public types ─────────────────────────────────────────────────────────────

export interface CanonicalizationResult {
  readonly simplified:  LayoutTransform[];
  /** How many transforms were eliminated versus the input. */
  readonly eliminated:  number;
  readonly description: string;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Simplify a chain of LayoutTransforms by composing adjacent pairs and
 * removing identities.  Returns a CanonicalizationResult with the simplified
 * chain and a human-readable description of what changed.
 */
export function simplifyTransformChain(
  transforms: readonly LayoutTransform[],
): CanonicalizationResult {
  if (transforms.length === 0) {
    return { simplified: [], eliminated: 0, description: "empty chain" };
  }

  const original = [...transforms];
  let   current  = [...transforms];
  let   changed  = true;

  while (changed) {
    changed = false;
    const next: LayoutTransform[] = [];
    let i = 0;

    while (i < current.length) {
      // Try to compose current[i] and current[i+1].
      if (i + 1 < current.length) {
        const composed = composeTransforms(current[i], current[i + 1]);
        if (composed !== null) {
          next.push(composed);
          i += 2;
          changed = true;
          continue;
        }
      }
      // Drop identity transforms when there are other transforms left.
      if (current[i].kind === "identity" && current.length > 1) {
        i++;
        changed = true;
        continue;
      }
      next.push(current[i]);
      i++;
    }

    current = next;
  }

  const eliminated  = original.length - current.length;
  const description = eliminated === 0
    ? "no simplification possible"
    : `${eliminated} transform(s) removed: ` +
      `[${original.map(formatTransform).join(", ")}] → [${current.map(formatTransform).join(", ")}]`;

  return { simplified: current, eliminated, description };
}

/**
 * Quick test: do two adjacent transforms cancel completely to an identity?
 * This is the precondition checked before a cancellation graph rewrite.
 */
export function transformsCancel(t1: LayoutTransform, t2: LayoutTransform): boolean {
  return areInverseTransforms(t1, t2);
}

/**
 * Try to simplify a pair of adjacent transforms into a single transform.
 * Returns null when composition is not possible or produces no simplification.
 */
export function simplifyPair(t1: LayoutTransform, t2: LayoutTransform): LayoutTransform | null {
  return composeTransforms(t1, t2);
}
