"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// passes/fusionPass.ts
//
// Operator Fusion Pass — the heart of the compiler.
//
// Algorithm
// ─────────
// 1. Clone the input graph (pass operates on the clone; original is untouched).
// 2. Run the PatternMatcher to find all non-overlapping fusible chains.
// 3. For each chain, ask the CostModel whether fusion is profitable.
// 4. For approved chains, apply the graph rewrite atomically:
//      a. Extract chain boundaries (external-inputs, external-outputs, internals).
//      b. Create the fused replacement Node + output Tensors.
//      c. Insert the fused node (positioned right after the last chain node).
//      d. Rewire downstream consumers to use the new fused output tensors.
//      e. Update graph-level output declarations.
//      f. Remove all original chain nodes.
//      g. Remove internal tensors and the now-replaced external output tensors.
// 5. Return the modified graph and a structured log of all fusions applied.
//
// Safety guarantees
// ─────────────────
// • The graph is cloned before modification → original always remains intact.
// • Chains are non-overlapping by matcher construction.
// • The CostModel double-checks every intermediate boundary.
// • After all fusions, the caller (PassManager) validates graph invariants.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.FusionPass = void 0;
const fusionAnalysis_1 = require("../analysis/fusionAnalysis");
const graphUtils_1 = require("../utils/graphUtils");
class FusionPass {
    constructor(registry, costModel) {
        this.registry = registry;
        this.costModel = costModel;
        this.name = "FusionPass";
    }
    run(graph) {
        const logs = [];
        const workGraph = graph.clone();
        // ── Step 1: Discover candidate chains (with diagnostics) ──────────────
        const analysis = (0, fusionAnalysis_1.analyzeFusion)(workGraph, this.registry.getRules());
        // Log per-node rejection reasons — this is the key improvement over
        // the old "No fusible chains found" single-line message.
        for (const rej of analysis.rejections) {
            const ruleName = rej.attemptedRule
                ? ` (rule: ${rej.attemptedRule.name ?? rej.attemptedRule.pattern.join("→")})`
                : "";
            logs.push({
                level: "info",
                message: `Rejected ${rej.nodeId}(${rej.op}) [${rej.reason}]${ruleName}: ${rej.detail}`,
            });
        }
        const chains = analysis.candidates;
        if (chains.length === 0) {
            logs.push({
                level: "info",
                message: `No fusible chains found (${analysis.stats.rejectedNodes} node(s) rejected).`,
            });
            return { graph, changed: false, logs };
        }
        logs.push({
            level: "info",
            message: `Found ${chains.length} candidate chain(s) (${analysis.stats.rejectedNodes} rejected).`,
        });
        let changed = false;
        // ── Step 2: Evaluate and apply each chain ─────────────────────────────
        for (const chain of chains) {
            if (!this.costModel.shouldFuse(chain, workGraph)) {
                const ops = chain.nodeIds.map(id => workGraph.getNode(id).op).join(" → ");
                const eval_ = this.costModel.evaluate(chain, workGraph);
                logs.push({
                    level: "info",
                    message: `Skipped [${ops}]: cost model rejected — ${eval_.reason}`,
                });
                continue;
            }
            const applied = this._applyFusion(workGraph, chain, logs);
            if (applied)
                changed = true;
        }
        if (!changed) {
            return { graph, changed: false, logs };
        }
        return { graph: workGraph, changed: true, logs };
    }
    // ─── Private: graph rewrite ──────────────────────────────────────────────
    _applyFusion(graph, chain, logs) {
        const consumers = (0, graphUtils_1.buildConsumerMap)(graph);
        const { externalInputs, externalOutputs, internalTensors } = (0, graphUtils_1.extractChainBoundary)(graph, chain.nodeIds, consumers);
        // Safety: a fused node must expose at least one output.
        if (externalOutputs.length === 0) {
            logs.push({
                level: "warn",
                message: `Chain [${chain.nodeIds.join(" → ")}] has no external outputs; skipping.`,
            });
            return false;
        }
        const fusedOp = chain.rule.fusedOp;
        const { node: fusedNode, tensors: fusedTensors } = (0, graphUtils_1.createFusedNode)(fusedOp, externalInputs, externalOutputs, graph);
        // Describe the fusion for the log.
        const chainDesc = chain.nodeIds
            .map(id => `${id}(${graph.getNode(id).op})`)
            .join(" → ");
        logs.push({
            level: "info",
            message: `Fusing: [${chainDesc}] ⟹  ${fusedNode.id}(${fusedOp})`,
        });
        logs.push({
            level: "info",
            message: `  Inputs:  [${externalInputs.join(", ")}]`,
        });
        logs.push({
            level: "info",
            message: `  Outputs: [${externalOutputs.join(", ")}] → [${fusedTensors.map(t => t.id).join(", ")}]`,
        });
        // ── Build old→new output tensor id mapping ─────────────────────────────
        const outputRemap = new Map();
        for (let i = 0; i < externalOutputs.length; i++) {
            outputRemap.set(externalOutputs[i], fusedTensors[i].id);
        }
        // ── a. Insert fused node right after the last chain node ───────────────
        graph._insertNode(fusedNode, fusedTensors, chain.nodeIds[chain.nodeIds.length - 1]);
        // ── b. Rewire downstream consumers ────────────────────────────────────
        for (const node of graph.nodes.values()) {
            if (node.id === fusedNode.id)
                continue;
            const newInputs = node.inputs.map(tid => outputRemap.get(tid) ?? tid);
            const rewired = newInputs.some((tid, i) => tid !== node.inputs[i]);
            if (rewired) {
                graph._replaceNode(node.id, { ...node, inputs: newInputs });
            }
        }
        // ── c. Update graph-level output declarations ──────────────────────────
        for (const [oldTid, newTid] of outputRemap) {
            graph._replaceOutputTensor(oldTid, newTid);
        }
        // ── d. Remove the original chain nodes ────────────────────────────────
        for (const nid of chain.nodeIds) {
            graph._removeNode(nid);
        }
        // ── e. Remove internal (now-orphaned) tensors ──────────────────────────
        for (const tid of internalTensors) {
            graph._removeTensor(tid);
        }
        // ── f. Remove the old external-output tensors (replaced by fused ones) ─
        for (const tid of externalOutputs) {
            graph._removeTensor(tid);
        }
        return true;
    }
}
exports.FusionPass = FusionPass;
//# sourceMappingURL=fusionPass.js.map