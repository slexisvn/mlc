// ─────────────────────────────────────────────────────────────────────────────
// ops/opContracts.ts
//
// Op-level semantic contracts used by layout analysis, fusion decision logic,
// and the pre-layout graph-simplification passes (CF, CSE, DCE).
//
// Each OpContract captures five independent facts about an operator:
//
//   fusibilityClass   — whether the op can be part of a fused kernel:
//                         "fusible"     -> always OK to fuse
//                         "conditional" -> may be fused depending on context
//                         "unfusible"   -> never fuse (e.g. split, concat)
//
//   layoutBehavior    — how the op interacts with tensor layout:
//                         "agnostic"    -> indifferent to layout (elementwise)
//                         "preserving"  -> propagates input layout to outputs
//                         "sensitive"   -> has required input/output layouts
//                         "transforming"-> explicitly changes the layout
//
//   requiredInputLayouts / outputLayout
//                     — for "sensitive" ops: accepted input formats and the
//                         resulting output format.
//
//   pure              — the op is side-effect-free: given the same input
//                         tensors it always produces bit-identical outputs.
//                         Required for CSE (value-numbering) and for
//                         ConstantFoldingPass to propagate constants.
//                         Default: false (conservative).
//
//   foldable          — the op can be evaluated at compile time by
//                         ConstantFoldingPass when *all* inputs carry a
//                         ConstantPayload.  Implies pure.
//                         Default: false.
//
// Extensibility
// Backends register their own contracts via OpContractRegistry.register().
// ─────────────────────────────────────────────────────────────────────────────

import { LayoutFormat, Layouts } from "../ir/layouts";

// --- Contract types ---

export type OpLayoutBehavior = "agnostic" | "preserving" | "sensitive" | "transforming";
export type FusibilityClass  = "fusible" | "unfusible" | "conditional";

export interface OpContract {
  readonly op:                    string;
  readonly fusibilityClass:       FusibilityClass;
  readonly layoutBehavior:        OpLayoutBehavior;
  readonly requiredInputLayouts?: readonly LayoutFormat[];
  readonly outputLayout?:         LayoutFormat;
  readonly pure?:                 boolean;
  readonly foldable?:             boolean;
  readonly description?:          string;
}

// --- Registry ---

export class OpContractRegistry {
  private readonly _contracts: Map<string, OpContract> = new Map();

  constructor(initial: readonly OpContract[] = []) {
    for (const c of initial) this._contracts.set(c.op, c);
  }

  register(contract: OpContract): this {
    this._contracts.set(contract.op, contract);
    return this;
  }

  get(op: string): OpContract | undefined { return this._contracts.get(op); }
  has(op: string): boolean                { return this._contracts.has(op); }

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
  isFusible(op: string): boolean {
    const c = this._contracts.get(op);
    if (c === undefined) return true;
    return c.fusibilityClass === "fusible" || c.fusibilityClass === "conditional";
  }
  isPure(op: string): boolean    { return this._contracts.get(op)?.pure    === true; }
  isFoldable(op: string): boolean { return this._contracts.get(op)?.foldable === true; }
  getAll(): readonly OpContract[] { return [...this._contracts.values()]; }
}

// --- Default built-in contracts ---

export const DEFAULT_OP_CONTRACTS: readonly OpContract[] = [
  { op: "add",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Element-wise add" },
  { op: "sub",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Element-wise subtract" },
  { op: "mul",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Element-wise multiply" },
  { op: "div",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Element-wise divide" },
  { op: "relu",     fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "ReLU activation" },
  { op: "sigmoid",  fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Sigmoid activation" },
  { op: "tanh",     fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Tanh activation" },
  { op: "gelu",     fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "GELU activation" },
  { op: "exp",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Exponential" },
  { op: "sqrt",     fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Square root" },
  { op: "neg",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Element-wise negate" },
  { op: "abs",      fusibilityClass: "fusible",     layoutBehavior: "agnostic",     pure: true,  foldable: true,  description: "Element-wise absolute value" },
  { op: "bn",       fusibilityClass: "fusible",     layoutBehavior: "preserving",   pure: true,  foldable: false, description: "Batch normalisation" },
  { op: "ln",       fusibilityClass: "fusible",     layoutBehavior: "preserving",   pure: true,  foldable: false, description: "Layer normalisation" },
  { op: "dropout",  fusibilityClass: "fusible",     layoutBehavior: "preserving",   pure: false, foldable: false, description: "Dropout (stochastic)" },
  { op: "conv",     fusibilityClass: "fusible",     layoutBehavior: "sensitive",    pure: true,  foldable: false, requiredInputLayouts: [Layouts.NCHW, Layouts.NHWC], description: "2-D convolution" },
  { op: "conv2d",   fusibilityClass: "fusible",     layoutBehavior: "sensitive",    pure: true,  foldable: false, requiredInputLayouts: [Layouts.NCHW, Layouts.NHWC], description: "2-D convolution (alias)" },
  { op: "matmul",   fusibilityClass: "fusible",     layoutBehavior: "sensitive",    pure: true,  foldable: false, requiredInputLayouts: [Layouts.NC], description: "Matrix multiplication" },
  { op: "gemm",     fusibilityClass: "fusible",     layoutBehavior: "sensitive",    pure: true,  foldable: false, requiredInputLayouts: [Layouts.NC], description: "General matrix multiply" },
  { op: "transpose",fusibilityClass: "conditional", layoutBehavior: "transforming", pure: true,  foldable: false, description: "Axis permutation; fromLayout/toLayout/perm stored in node attrs" },
  { op: "reshape",  fusibilityClass: "conditional", layoutBehavior: "transforming", pure: true,  foldable: false, description: "Shape reshape (may break layout semantics)" },
  { op: "split",    fusibilityClass: "unfusible",   layoutBehavior: "preserving",   pure: true,  foldable: false, description: "Tensor split (multiple outputs)" },
  { op: "concat",   fusibilityClass: "unfusible",   layoutBehavior: "preserving",   pure: true,  foldable: false, description: "Tensor concatenation" },
];

export const DEFAULT_CONTRACT_REGISTRY = new OpContractRegistry(DEFAULT_OP_CONTRACTS);