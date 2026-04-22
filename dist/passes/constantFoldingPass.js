"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// passes/constantFoldingPass.ts
//
// Compile-time constant propagation on the graph IR.
//
// Algorithm
// ─────────
// Walk the graph in topological order (graph.nodeOrder is topologically sorted
// for any valid DAG).  For each node:
//   1. Skip if the op is not marked `foldable` in the contract registry.
//   2. Skip if any input tensor lacks a `ConstantPayload`.
//   3. Evaluate the op numerically.
//   4. Attach a `ConstantPayload` to the single output tensor via
//      `graph._setConstantPayload`.
//   5. Replace the original compute node with a "const" pseudo-node (no inputs)
//      via `graph._replaceWithConstNode`, so that upstream producers of the
//      now-consumed inputs can be pruned by DeadCodeEliminationPass.
//
// Supported ops and broadcast semantics
// ──────────────────────────────────────
// Unary  (1 input):  relu, sigmoid, tanh, gelu, exp, sqrt, neg, abs
// Binary (2 inputs): add, sub, mul, div
//
// Binary broadcast rules (evaluated in order):
//   1. Same shape               → element-wise application
//   2. Operand A is a scalar    → broadcast A across B's elements
//   3. Operand B is a scalar    → broadcast B across A's elements
//   4. All other shape combos   → log warn and skip (no fold)
//
// Only single-output foldable nodes are handled in this iteration.
//
// Invariants preserved
// ────────────────────
// • SSA: each output tensor retains exactly one producer (the new const node).
// • No dangling edges: the const node carries no input edges.
// • graph.nodeOrder remains topologically ordered.
// • validateGraph() passes after this pass.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConstantFoldingPass = void 0;
exports.resetCFCounter = resetCFCounter;
const opContracts_1 = require("../ops/opContracts");
// ─── Module-level id counter ─────────────────────────────────────────────────
let _cfCounter = 0;
/** Reset the const-node id counter (for deterministic test output). */
function resetCFCounter() { _cfCounter = 0; }
// ─── Numeric kernels ─────────────────────────────────────────────────────────
/** Total element count for `shape`.  Scalars (shape = []) have count 1. */
function elementCount(shape) {
    return shape.length === 0 ? 1 : shape.reduce((acc, d) => acc * d, 1);
}
function applyUnary(op, x) {
    switch (op) {
        case "relu": return x > 0 ? x : 0;
        case "sigmoid": return 1 / (1 + Math.exp(-x));
        case "tanh": return Math.tanh(x);
        // Tanh-based GELU approximation matching PyTorch's default implementation.
        case "gelu": return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        case "exp": return Math.exp(x);
        case "sqrt": return Math.sqrt(x);
        case "neg": return -x;
        case "abs": return Math.abs(x);
        default: throw new Error(`ConstantFoldingPass: no unary kernel for op "${op}"`);
    }
}
function applyBinary(op, a, b) {
    switch (op) {
        case "add": return a + b;
        case "sub": return a - b;
        case "mul": return a * b;
        case "div": return a / b;
        default: throw new Error(`ConstantFoldingPass: no binary kernel for op "${op}"`);
    }
}
// ─── Evaluation dispatcher ────────────────────────────────────────────────────
/**
 * Attempt to evaluate `node` given its pre-computed input constant payloads.
 *
 * Returns a flat result array and the output shape on success.
 * Returns `null` if the op cannot be folded (unsupported arity or shape mismatch).
 */
function tryFold(node, payloads, graph, logs) {
    const UNARY_OPS = new Set(["relu", "sigmoid", "tanh", "gelu", "exp", "sqrt", "neg", "abs"]);
    const BINARY_OPS = new Set(["add", "sub", "mul", "div"]);
    // ── Unary ──────────────────────────────────────────────────────────────────
    if (UNARY_OPS.has(node.op)) {
        if (node.inputs.length !== 1) {
            logs.push({ level: "warn", message: `CF: "${node.op}" expected 1 input, got ${node.inputs.length} — skipping` });
            return null;
        }
        const inTensor = graph.getTensor(node.inputs[0]);
        const inData = payloads[0].data;
        return {
            data: inData.map(x => applyUnary(node.op, x)),
            shape: [...inTensor.shape],
        };
    }
    // ── Binary ─────────────────────────────────────────────────────────────────
    if (BINARY_OPS.has(node.op)) {
        if (node.inputs.length !== 2) {
            logs.push({ level: "warn", message: `CF: "${node.op}" expected 2 inputs, got ${node.inputs.length} — skipping` });
            return null;
        }
        const tA = graph.getTensor(node.inputs[0]);
        const tB = graph.getTensor(node.inputs[1]);
        const dataA = payloads[0].data;
        const dataB = payloads[1].data;
        // Case 1: identical shapes → element-wise.
        const sameShape = tA.shape.length === tB.shape.length &&
            tA.shape.every((d, i) => d === tB.shape[i]);
        if (sameShape) {
            return {
                data: dataA.map((a, i) => applyBinary(node.op, a, dataB[i])),
                shape: [...tA.shape],
            };
        }
        // Case 2: A is scalar → broadcast.
        if (elementCount(tA.shape) === 1) {
            const scalar = dataA[0];
            return {
                data: dataB.map(b => applyBinary(node.op, scalar, b)),
                shape: [...tB.shape],
            };
        }
        // Case 3: B is scalar → broadcast.
        if (elementCount(tB.shape) === 1) {
            const scalar = dataB[0];
            return {
                data: dataA.map(a => applyBinary(node.op, a, scalar)),
                shape: [...tA.shape],
            };
        }
        // Incompatible shapes — skip.
        logs.push({
            level: "warn",
            message: `CF: "${node.op}" shape mismatch [${tA.shape}] vs [${tB.shape}] — skipping`,
        });
        return null;
    }
    // Op registered as foldable but no kernel implemented (should not happen with
    // the default registry, but safe to handle for custom ops).
    logs.push({ level: "warn", message: `CF: no kernel implemented for op "${node.op}" — skipping` });
    return null;
}
// ─── Pass ─────────────────────────────────────────────────────────────────────
class ConstantFoldingPass {
    constructor(opRegistry = opContracts_1.DEFAULT_CONTRACT_REGISTRY) {
        this.opRegistry = opRegistry;
        this.name = "ConstantFoldingPass";
    }
    run(graph) {
        const logs = [];
        const workGraph = graph.clone();
        let folded = 0;
        let skipped = 0;
        // Snapshot the node order before iteration; _replaceWithConstNode mutates
        // the order list in-place but preserves existing indices, so snapshotting
        // is a conservative choice for determinism.
        const order = [...workGraph.nodeOrder];
        for (const nodeId of order) {
            // Nodes can theoretically be removed by other mechanisms in future
            // iterations; guard defensively.
            const node = workGraph.nodes.get(nodeId);
            if (!node)
                continue;
            // ── Gate 1: contract marks this op as foldable ────────────────────────
            if (!this.opRegistry.isFoldable(node.op))
                continue;
            // ── Gate 2: single-output only in this iteration ──────────────────────
            if (node.outputs.length !== 1) {
                logs.push({
                    level: "info",
                    message: `CF: skipping multi-output node "${nodeId}" (op=${node.op})`,
                });
                skipped++;
                continue;
            }
            // ── Gate 3: every input must carry a ConstantPayload ──────────────────
            const inputPayloads = node.inputs.map(tid => workGraph.getTensor(tid).constantPayload);
            if (inputPayloads.some(p => p === undefined))
                continue;
            // ── Evaluate the operation numerically ────────────────────────────────
            const result = tryFold(node, inputPayloads, workGraph, logs);
            if (result === null) {
                skipped++;
                continue;
            }
            // ── Attach the computed constant to the output tensor ─────────────────
            workGraph._setConstantPayload(node.outputs[0], { data: result.data });
            // ── Swap compute node for a source "const" node (no inputs) ──────────
            const constNodeId = `n_cf_${_cfCounter++}`;
            workGraph._replaceWithConstNode(nodeId, constNodeId, { foldedFrom: node.op });
            logs.push({
                level: "info",
                message: `CF: folded "${nodeId}" (op=${node.op}) → const "${constNodeId}", ` +
                    `output shape=[${result.shape.join(",")}]`,
            });
            folded++;
        }
        logs.push({
            level: "info",
            message: `ConstantFoldingPass complete: ${folded} node(s) folded, ${skipped} skipped.`,
        });
        return folded > 0
            ? { graph: workGraph, changed: true, logs }
            : { graph, changed: false, logs };
    }
}
exports.ConstantFoldingPass = ConstantFoldingPass;
//# sourceMappingURL=constantFoldingPass.js.map