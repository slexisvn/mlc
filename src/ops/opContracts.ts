// ─────────────────────────────────────────────────────────────────────────────
// ops/opContracts.ts
//
// Op-level semantic contracts used by layout analysis and fusion decision logic.
//
// Each OpContract captures three independent facts about an operator:
//
//   fusibilityClass   — whether the op can be part of a fused kernel:
//                         "fusible"     → always OK to fuse
//                         "conditional" → may be fused depending on context
//                         "unfusible"   → never fuse (e.g. split, concat)
//
//   layoutBehavior    — how the op interacts with tensor layout:
//                         "agnostic"    → indifferent to layout (elementwise)
//                         "preserving"  → propagates input layout to outputs
//                         "sensitive"   → has required input/output layouts
//                         "transforming"→ explicitly changes the layout
//
//   requiredInputLayouts / outputLayout
//                     — for "sensitive" ops: the set of accepted input formats
//                         and the resulting output format.
//
// Extensibility
// ─────────────
// Backends register their own contracts via OpContractRegistry.register().
// The analysis layer falls back to conservative "agnostic + fusible" defaults
// for any op not in the registry — so adding a new backend op requires only
// registering its contract; no other code changes are needed.
// ─────────────────────────────────────────────────────────────────────────────

import { LayoutFormat, Layouts } from "../ir/layouts";

// ─── Contract types ───────────────────────────────────────────────────────────

export type OpLayoutBehavior = "agnostic" | "preserving" | "sensitive" | "transforming";
export type FusibilityClass  = "fusible" | "unfusible" | "conditional";

export interface OpContract {
  readonly op:                    string;
  readonly fusibilityClass:       FusibilityClass;
  readonly layoutBehavior:        OpLayoutBehavior;
  /** Formats the op can accept as primary input (checked during layout analysis). */
  readonly requiredInputLayouts?: readonly LayoutFormat[];
  /** Layout of the op's output when inputs are in a required format. */
  readonly outputLayout?:         LayoutFormat;
  /** Human-readable description shown in diagnostic output. */
  readonly description?:          string;
}

// ─── Registry ────────────────────────────────────────────────────────────────

export class OpContractRegistry {
  private readonly _contracts: Map<string, OpContract> = new Map();

  constructor(initial: readonly OpContract[] = []) {
    for (const c of initial) this._contracts.set(c.op, c);
  }

  /** Add or overwrite a contract. Returns `this` for chaining. */
  register(contract: OpContract): this {
    this._contracts.set(contract.op, contract);
    return this;
  }

  get(op: string): OpContract | undefined {
    return this._contracts.get(op);
  }

  has(op: string): boolean {
    return this._contracts.has(op);
  }

  /**
   * True when the op has no layout requirements — either it is not registered
   * (optimistic open-world default) or its contract says "agnostic".
   */
  isLayoutAgnostic(op: string): boolean {
    const c = this._contracts.get(op);
    return c === undefined || c.layoutBehavior === "agnostic";
  }

  isLayoutSensitive(op: string): boolean {
    const c = this._contracts.get(op);
    return c !== undefined && c.layoutBehavior === "sensitive";
  }

  isLayoutTransforming(op: string): boolean {
    const c = this._contracts.get(op);
    return c !== undefined && c.layoutBehavior === "transforming";
  }

  isLayoutPreserving(op: string): boolean {
    const c = this._contracts.get(op);
    return c !== undefined && c.layoutBehavior === "preserving";
  }

  /**
   * True when the op can participate in a fused kernel.
   * Ops not in the registry are treated as fusible (optimistic default).
   */
  isFusible(op: string): boolean {
    const c = this._contracts.get(op);
    if (c === undefined) return true;
    return c.fusibilityClass === "fusible" || c.fusibilityClass === "conditional";
  }

  getAll(): readonly OpContract[] {
    return [...this._contracts.values()];
  }
}

// ─── Default built-in contracts ───────────────────────────────────────────────

export const DEFAULT_OP_CONTRACTS: readonly OpContract[] = [
  // ── Elementwise — layout agnostic ────────────────────────────────────────
  { op: "add",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "Element-wise add" },
  { op: "sub",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "Element-wise subtract" },
  { op: "mul",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "Element-wise multiply" },
  { op: "div",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "Element-wise divide" },
  { op: "relu",     fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "ReLU activation" },
  { op: "sigmoid",  fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "Sigmoid activation" },
  { op: "tanh",     fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "Tanh activation" },
  { op: "gelu",     fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "GELU activation" },
  { op: "exp",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "Exponential" },
  { op: "sqrt",     fusibilityClass: "fusible",     layoutBehavior: "agnostic",     description: "Square root" },
  // ── Normalisation — layout preserving ────────────────────────────────────
  { op: "bn",       fusibilityClass: "fusible",     layoutBehavior: "preserving",   description: "Batch normalisation" },
  { op: "ln",       fusibilityClass: "fusible",     layoutBehavior: "preserving",   description: "Layer normalisation" },
  { op: "dropout",  fusibilityClass: "fusible",     layoutBehavior: "preserving",   description: "Dropout" },
  // ── Convolutions — layout sensitive ─────────────────────────────────────
  {
    op:                   "conv",
    fusibilityClass:      "fusible",
    layoutBehavior:       "sensitive",
    requiredInputLayouts: [Layouts.NCHW, Layouts.NHWC],
    description:          "2-D convolution",
  },
  {
    op:                   "conv2d",
    fusibilityClass:      "fusible",
    layoutBehavior:       "sensitive",
    requiredInputLayouts: [Layouts.NCHW, Layouts.NHWC],
    description:          "2-D convolution (alias)",
  },
  // ── Linear / matmul — layout sensitive ──────────────────────────────────
  {
    op:                   "matmul",
    fusibilityClass:      "fusible",
    layoutBehavior:       "sensitive",
    requiredInputLayouts: [Layouts.NC],
    description:          "Matrix multiplication",
  },
  {
    op:                   "gemm",
    fusibilityClass:      "fusible",
    layoutBehavior:       "sensitive",
    requiredInputLayouts: [Layouts.NC],
    description:          "General matrix multiply",
  },
  // ── Layout-transforming ops ──────────────────────────────────────────────
  {
    op:              "transpose",
    fusibilityClass: "conditional",
    layoutBehavior:  "transforming",
    description:     "Axis permutation; fromLayout/toLayout/perm stored in node attrs",
  },
  {
    op:              "reshape",
    fusibilityClass: "conditional",
    layoutBehavior:  "transforming",
    description:     "Shape reshape (may break layout semantics)",
  },
  // ── Non-fusible ops ──────────────────────────────────────────────────────
  { op: "split",    fusibilityClass: "unfusible",   layoutBehavior: "preserving",   description: "Tensor split (multiple outputs)" },
  { op: "concat",   fusibilityClass: "unfusible",   layoutBehavior: "preserving",   description: "Tensor concatenation" },
];

/** Singleton registry pre-populated with all DEFAULT_OP_CONTRACTS. */
export const DEFAULT_CONTRACT_REGISTRY = new OpContractRegistry(DEFAULT_OP_CONTRACTS);
