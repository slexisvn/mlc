// ─────────────────────────────────────────────────────────────────────────────
// patterns/rules.ts
//
// Fusion rule registry.
//
// Design rationale for extensibility:
//   Rules are pure data (FusionRule), completely decoupled from matching logic.
//   Adding a new rule:
//     registry.addRule({ pattern: ["gelu"], fusedOp: "fast_gelu" })
//   …or extending the defaults before constructing the pipeline.
//   No other code needs to change.
//
// Rule ordering matters only for greedy longest-match: the PatternMatcher
// tries longer rules first, so a 3-op rule takes priority over any 2-op prefix.
// ─────────────────────────────────────────────────────────────────────────────

import { FusionRule } from "../ir/types";

/**
 * Built-in fusion rules covering the most common DNN patterns.
 * Ordered from longest to shortest within semantic groups so the registry
 * already provides a reasonable greedy-match priority baseline.
 */
export const DEFAULT_FUSION_RULES: readonly FusionRule[] = [
  // ── 3-op chains ───────────────────────────────────────────────────────────
  { pattern: ["conv",   "bn",  "relu"],  fusedOp: "conv_bn_relu"   },
  { pattern: ["matmul", "add", "relu"],  fusedOp: "linear_relu"    },

  // ── 2-op chains ───────────────────────────────────────────────────────────
  { pattern: ["conv",   "relu"],         fusedOp: "conv_relu"      },
  { pattern: ["conv",   "bn"],           fusedOp: "conv_bn"        },
  { pattern: ["matmul", "relu"],         fusedOp: "matmul_relu"    },
  { pattern: ["matmul", "add"],          fusedOp: "linear"         },
  { pattern: ["add",    "relu"],         fusedOp: "add_relu"       },
  { pattern: ["add",    "sigmoid"],      fusedOp: "add_sigmoid"    },
  { pattern: ["mul",    "relu"],         fusedOp: "mul_relu"       },
];

/**
 * Mutable registry of fusion rules.
 *
 * Pass an instance to RuleRegistry to the FusionPass so user code can extend
 * the rule set without touching any pass internals.
 */
export class RuleRegistry {
  private readonly _rules: FusionRule[];

  constructor(initialRules: readonly FusionRule[] = DEFAULT_FUSION_RULES) {
    this._rules = [...initialRules];
  }

  /** Add a new fusion rule at the end of the registry. */
  addRule(rule: FusionRule): this {
    this._rules.push(rule);
    return this;
  }

  /** Read-only view of all registered rules. */
  getRules(): readonly FusionRule[] {
    return this._rules;
  }

  /** Number of rules currently registered. */
  get size(): number {
    return this._rules.length;
  }
}
