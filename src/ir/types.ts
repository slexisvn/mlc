// ─────────────────────────────────────────────────────────────────────────────
// ir/types.ts
//
// Shared primitive types used across every layer of the compiler.
// No runtime logic here; pure type aliases and interfaces.
// ─────────────────────────────────────────────────────────────────────────────

/** Element data type of a tensor. Open-ended string allows backends to extend. */
export type DType = "float32" | "float64" | "int32" | "int64" | "bool" | string;

/** Tensor shape: ordered list of dimension sizes.  -1 = dynamic dimension. */
export type Shape = number[];

/** Generic attribute bag attached to a Node (e.g. stride, padding, axis). */
export type Attrs = Record<string, unknown>;

/**
 * A fusion rule maps a linear sequence of op names to a single fused op.
 *
 * Example:
 *   { pattern: ["add", "relu"], fusedOp: "add_relu" }
 *
 * Rules are data-only; the FusionPass drives the actual rewrite.
 * Adding a new rule requires only appending to the registry — zero code changes
 * to any other component.
 */
export interface FusionRule {
  /** Ordered op names that must appear as a linear chain, e.g. ["add", "relu"]. */
  readonly pattern: string[];
  /** Name of the fused operator that replaces the matched chain. */
  readonly fusedOp: string;
  /** Human-readable rule name, used in diagnostic output. */
  readonly name?: string;
}

/**
 * A candidate fusion chain discovered by the analysis layer.
 * Structural minimum accepted by CostModel and analysis utilities.
 */
export interface ChainCandidate {
  readonly rule:    FusionRule;
  readonly nodeIds: readonly string[];
}
