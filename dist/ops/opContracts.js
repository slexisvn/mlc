"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_CONTRACT_REGISTRY = exports.DEFAULT_OP_CONTRACTS = exports.OpContractRegistry = void 0;
const layouts_1 = require("../ir/layouts");
// --- Registry ---
class OpContractRegistry {
    constructor(initial = []) {
        this._contracts = new Map();
        for (const c of initial)
            this._contracts.set(c.op, c);
    }
    register(contract) {
        this._contracts.set(contract.op, contract);
        return this;
    }
    get(op) { return this._contracts.get(op); }
    has(op) { return this._contracts.has(op); }
    isLayoutAgnostic(op) {
        const c = this._contracts.get(op);
        return c === undefined || c.layoutBehavior === "agnostic";
    }
    isLayoutSensitive(op) {
        const c = this._contracts.get(op);
        return c !== undefined && c.layoutBehavior === "sensitive";
    }
    isLayoutTransforming(op) {
        const c = this._contracts.get(op);
        return c !== undefined && c.layoutBehavior === "transforming";
    }
    isLayoutPreserving(op) {
        const c = this._contracts.get(op);
        return c !== undefined && c.layoutBehavior === "preserving";
    }
    isFusible(op) {
        const c = this._contracts.get(op);
        if (c === undefined)
            return true;
        return c.fusibilityClass === "fusible" || c.fusibilityClass === "conditional";
    }
    isPure(op) { return this._contracts.get(op)?.pure === true; }
    isFoldable(op) { return this._contracts.get(op)?.foldable === true; }
    getAll() { return [...this._contracts.values()]; }
}
exports.OpContractRegistry = OpContractRegistry;
// --- Default built-in contracts ---
exports.DEFAULT_OP_CONTRACTS = [
    { op: "add", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Element-wise add" },
    { op: "sub", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Element-wise subtract" },
    { op: "mul", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Element-wise multiply" },
    { op: "div", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Element-wise divide" },
    { op: "relu", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "ReLU activation" },
    { op: "sigmoid", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Sigmoid activation" },
    { op: "tanh", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Tanh activation" },
    { op: "gelu", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "GELU activation" },
    { op: "exp", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Exponential" },
    { op: "sqrt", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Square root" },
    { op: "neg", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Element-wise negate" },
    { op: "abs", fusibilityClass: "fusible", layoutBehavior: "agnostic", pure: true, foldable: true, description: "Element-wise absolute value" },
    { op: "bn", fusibilityClass: "fusible", layoutBehavior: "preserving", pure: true, foldable: false, description: "Batch normalisation" },
    { op: "ln", fusibilityClass: "fusible", layoutBehavior: "preserving", pure: true, foldable: false, description: "Layer normalisation" },
    { op: "dropout", fusibilityClass: "fusible", layoutBehavior: "preserving", pure: false, foldable: false, description: "Dropout (stochastic)" },
    { op: "conv", fusibilityClass: "fusible", layoutBehavior: "sensitive", pure: true, foldable: false, requiredInputLayouts: [layouts_1.Layouts.NCHW, layouts_1.Layouts.NHWC], description: "2-D convolution" },
    { op: "conv2d", fusibilityClass: "fusible", layoutBehavior: "sensitive", pure: true, foldable: false, requiredInputLayouts: [layouts_1.Layouts.NCHW, layouts_1.Layouts.NHWC], description: "2-D convolution (alias)" },
    { op: "matmul", fusibilityClass: "fusible", layoutBehavior: "sensitive", pure: true, foldable: false, requiredInputLayouts: [layouts_1.Layouts.NC], description: "Matrix multiplication" },
    { op: "gemm", fusibilityClass: "fusible", layoutBehavior: "sensitive", pure: true, foldable: false, requiredInputLayouts: [layouts_1.Layouts.NC], description: "General matrix multiply" },
    { op: "transpose", fusibilityClass: "conditional", layoutBehavior: "transforming", pure: true, foldable: false, description: "Axis permutation; fromLayout/toLayout/perm stored in node attrs" },
    { op: "reshape", fusibilityClass: "conditional", layoutBehavior: "transforming", pure: true, foldable: false, description: "Shape reshape (may break layout semantics)" },
    { op: "split", fusibilityClass: "unfusible", layoutBehavior: "preserving", pure: true, foldable: false, description: "Tensor split (multiple outputs)" },
    { op: "concat", fusibilityClass: "unfusible", layoutBehavior: "preserving", pure: true, foldable: false, description: "Tensor concatenation" },
];
exports.DEFAULT_CONTRACT_REGISTRY = new OpContractRegistry(exports.DEFAULT_OP_CONTRACTS);
//# sourceMappingURL=opContracts.js.map