"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// optimizer/costModel.ts
//
// Simple heuristic cost model for fusion profitability.
//
// Philosophy
// ──────────
// A cost model answers one question: "Is fusing this chain worth it?"
// The current model is intentionally conservative and transparent:
//
//   Rule 1  Chain length ≥ 2 (trivially ensured by the matcher, but guarded here).
//   Rule 2  Chain length ≤ configurable maximum (prevents monster kernels).
//   Rule 3  Node count strictly decreases (N nodes → 1 node: always true for N ≥ 2).
//   Rule 4  No intermediate tensor escapes the chain (double-check of matcher invariant).
//
// Future extensions:
//   • Memory-bandwidth model (weight loads vs element-wise arithmetic).
//   • Backend-specific kernel availability table.
//   • Profile-guided costs from hardware counters.
//
// The interface is deliberately simple so a custom CostModel implementation
// can be swapped in by passing it to FusionPass.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.CostModel = exports.DEFAULT_COST_MODEL_CONFIG = void 0;
const graphUtils_1 = require("../utils/graphUtils");
exports.DEFAULT_COST_MODEL_CONFIG = {
    maxChainLength: 8,
};
class CostModel {
    constructor(config = {}) {
        this.config = { ...exports.DEFAULT_COST_MODEL_CONFIG, ...config };
    }
    /**
     * Decide whether fusing the given chain is profitable and safe.
     *
     * @param chain  A matched chain candidate (already validated by the matcher).
     * @param graph  The current graph (used for double-checking consumer invariants).
     * @returns true if the fusion should proceed.
     */
    shouldFuse(chain, graph) {
        return this.evaluate(chain, graph).shouldFuse;
    }
    /**
     * Return a structured evaluation with both the decision and the reason.
     * Accepts any ChainCandidate (MatchedChain is a subtype) for composability
     * with the analysis layer.
     */
    evaluate(chain, graph) {
        const n = chain.nodeIds.length;
        // Rule 1: Minimum chain length.
        if (n < 2) {
            return { shouldFuse: false, nodeCountReduction: 0, reason: "Chain too short (minimum 2 nodes)." };
        }
        // Rule 2: Maximum chain length.
        if (n > this.config.maxChainLength) {
            return {
                shouldFuse: false,
                nodeCountReduction: n - 1,
                reason: `Chain length ${n} exceeds maximum ${this.config.maxChainLength}.`,
            };
        }
        // Rule 3: Node-count reduction must be strictly positive.
        const nodeCountReduction = n - 1;
        if (nodeCountReduction < 1) {
            return { shouldFuse: false, nodeCountReduction, reason: "No node-count reduction achievable." };
        }
        // Rule 4: Belt-and-suspenders check — intermediate nodes must have
        //         single-consumer outputs that are not graph outputs.
        const consumers = (0, graphUtils_1.buildConsumerMap)(graph);
        const graphOutputSet = new Set(graph.outputIds);
        for (let i = 0; i < chain.nodeIds.length - 1; i++) {
            const node = graph.getNode(chain.nodeIds[i]);
            for (const tid of node.outputs) {
                if (graphOutputSet.has(tid)) {
                    return {
                        shouldFuse: false,
                        nodeCountReduction,
                        reason: `Intermediate tensor "${tid}" is a graph output.`,
                    };
                }
                const cons = consumers.get(tid) ?? [];
                if (cons.length !== 1) {
                    return {
                        shouldFuse: false,
                        nodeCountReduction,
                        reason: `Intermediate tensor "${tid}" has ${cons.length} consumer(s).`,
                    };
                }
            }
        }
        return {
            shouldFuse: true,
            nodeCountReduction,
            reason: `Fusing ${n} nodes → 1 saves ${nodeCountReduction} node(s).`,
        };
    }
}
exports.CostModel = CostModel;
//# sourceMappingURL=costModel.js.map