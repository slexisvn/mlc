// ─────────────────────────────────────────────────────────────────────────────
// patterns/layoutRules.ts
//
// Layout rewrite rules — pure data, decoupled from matching and rewrite logic.
//
// A LayoutRewriteRule describes a linear chain of op names whose layout
// transforms can be simplified by the LayoutTransformPass.  Two rule kinds:
//
//   "cancellation"  — the chain is a back-to-back pair of inverse transforms
//                     (e.g. NCHW→NHWC → NHWC→NCHW).  Both nodes are removed.
//
//   "propagation"   — the chain is a transforming op, any number of
//                     layout-agnostic ops, then the inverse transforming op
//                     (e.g. transpose → add → transpose).
//                     Both outer transposes are removed; the middle op runs in
//                     the original layout.
//
// Priority
// ─────────
// Higher-priority rules are tried first so longer/more-specific patterns win
// over shorter/less-specific ones.
// ─────────────────────────────────────────────────────────────────────────────

export type LayoutRuleKind = "cancellation" | "propagation" | "normalization";

export interface LayoutRewriteRule {
  readonly name:        string;
  readonly kind:        LayoutRuleKind;
  /** Ordered op names forming the pattern, e.g. ["transpose", "add", "transpose"]. */
  readonly pattern:     string[];
  readonly description: string;
  /** Higher value = tried before lower-priority rules. Default: 50. */
  readonly priority:    number;
}

// ─── Default layout rules ─────────────────────────────────────────────────────

export const DEFAULT_LAYOUT_RULES: readonly LayoutRewriteRule[] = [
  // ── Direct transpose–transpose cancellation (priority 100) ──────────────
  {
    name:        "transpose_transpose_cancel",
    kind:        "cancellation",
    pattern:     ["transpose", "transpose"],
    description: "Two consecutive inverse transposes cancel each other.",
    priority:    100,
  },
  // ── Propagation: transpose around elementwise binary ops (priority 80) ───
  {
    name:        "transpose_add_transpose_cancel",
    kind:        "propagation",
    pattern:     ["transpose", "add", "transpose"],
    description: "Hoist layout conversion out of an add.",
    priority:    80,
  },
  {
    name:        "transpose_sub_transpose_cancel",
    kind:        "propagation",
    pattern:     ["transpose", "sub", "transpose"],
    description: "Hoist layout conversion out of a subtract.",
    priority:    80,
  },
  {
    name:        "transpose_mul_transpose_cancel",
    kind:        "propagation",
    pattern:     ["transpose", "mul", "transpose"],
    description: "Hoist layout conversion out of a multiply.",
    priority:    80,
  },
  // ── Propagation: transpose around activations (priority 70) ─────────────
  {
    name:        "transpose_relu_transpose_cancel",
    kind:        "propagation",
    pattern:     ["transpose", "relu", "transpose"],
    description: "Hoist layout conversion out of a relu.",
    priority:    70,
  },
  {
    name:        "transpose_sigmoid_transpose_cancel",
    kind:        "propagation",
    pattern:     ["transpose", "sigmoid", "transpose"],
    description: "Hoist layout conversion out of a sigmoid.",
    priority:    70,
  },
  {
    name:        "transpose_tanh_transpose_cancel",
    kind:        "propagation",
    pattern:     ["transpose", "tanh", "transpose"],
    description: "Hoist layout conversion out of a tanh.",
    priority:    70,
  },
  {
    name:        "transpose_gelu_transpose_cancel",
    kind:        "propagation",
    pattern:     ["transpose", "gelu", "transpose"],
    description: "Hoist layout conversion out of a gelu.",
    priority:    70,
  },
];

// ─── Registry ─────────────────────────────────────────────────────────────────

export class LayoutRuleRegistry {
  private readonly _rules: LayoutRewriteRule[];

  constructor(initial: readonly LayoutRewriteRule[] = DEFAULT_LAYOUT_RULES) {
    this._rules = [...initial].sort((a, b) => b.priority - a.priority);
  }

  /** Add a rule and re-sort by priority. Returns `this` for chaining. */
  addRule(rule: LayoutRewriteRule): this {
    this._rules.push(rule);
    this._rules.sort((a, b) => b.priority - a.priority);
    return this;
  }

  getRules(): readonly LayoutRewriteRule[] {
    return this._rules;
  }

  get size(): number {
    return this._rules.length;
  }
}
