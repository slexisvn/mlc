"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// passes/deadCodeEliminationPass.ts
//
// Dead Code Elimination (DCE) on the graph IR.
//
// Algorithm
// ─────────
// DCE identifies nodes that do not contribute to any graph output and removes
// them along with the tensors they produce.
//
// Liveness is computed with a backward BFS from the graph's declared outputs:
//
//   1. Seed the live-tensor set with every graph output tensor id.
//   2. For each live tensor, mark its producer node as live.
//   3. For each newly live node, mark all of its input tensors as live.
//   4. Repeat until no new tensors or nodes are added (BFS fixpoint).
//
// After the BFS:
//   • Any node NOT in the live-node set is unreachable from the graph's
//     observable outputs and is therefore dead.
//   • Remove every dead node from the graph.
//   • Remove every tensor produced by a dead node (the tensor's producerNodeId
//     is non-null and points to a dead node).
//
// Graph input tensors (producerNodeId === null) are never removed, even if they
// are not consumed by any live node: they represent external feeds whose
// removal would silently change the graph's calling convention.  A warning is
// logged for unused graph inputs so the caller can inspect them.
//
// Interaction with ConstantFoldingPass
// ─────────────────────────────────────
// CF replaces compute nodes with "const" source nodes (no inputs).  After CF,
// the original producers of the constant inputs have no live consumers and are
// therefore dead.  DCE discovers and removes them, creating a cascade that
// prunes entire constant-only subgraphs.
//
// Invariants preserved
// ────────────────────
// • SSA: only live tensors remain; each has exactly one producer or is a graph
//        input.
// • No dangling edges: dead tensors are removed before any node that consumes
//        them could reference them (those consumer nodes are dead themselves).
// • graph.nodeOrder contains only live nodes after the pass.
// • validateGraph() passes after this pass.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.DeadCodeEliminationPass = void 0;
class DeadCodeEliminationPass {
    constructor() {
        this.name = "DeadCodeEliminationPass";
    }
    run(graph) {
        const logs = [];
        const workGraph = graph.clone();
        // ── Phase 1: backward BFS to mark live tensors and nodes ─────────────────
        const liveTensors = new Set();
        const liveNodes = new Set();
        // Seed the BFS with the graph's declared output tensors.
        const queue = [...workGraph.outputIds];
        for (const tid of queue)
            liveTensors.add(tid);
        while (queue.length > 0) {
            const tid = queue.shift();
            const tensor = workGraph.tensors.get(tid);
            if (!tensor)
                continue;
            // Graph-input tensors have no producer — nothing further to mark.
            if (tensor.producerNodeId === null)
                continue;
            const producerId = tensor.producerNodeId;
            if (liveNodes.has(producerId))
                continue; // already processed
            const producerNode = workGraph.nodes.get(producerId);
            if (!producerNode)
                continue;
            liveNodes.add(producerId);
            // Mark the node's input tensors as live and enqueue newly discovered ones.
            for (const inputTid of producerNode.inputs) {
                if (!liveTensors.has(inputTid)) {
                    liveTensors.add(inputTid);
                    queue.push(inputTid);
                }
            }
        }
        // ── Phase 2: remove dead nodes and their produced tensors ─────────────────
        let removedNodes = 0;
        let removedTensors = 0;
        // Collect all nodes absent from the live set.
        const deadNodeIds = [...workGraph.nodes.keys()].filter(id => !liveNodes.has(id));
        for (const nodeId of deadNodeIds) {
            const node = workGraph.nodes.get(nodeId);
            // Remove output tensors produced by this dead node first so no dangling
            // producerNodeId references remain.
            for (const tid of node.outputs) {
                workGraph._removeTensor(tid);
                removedTensors++;
            }
            workGraph._removeNode(nodeId);
            removedNodes++;
            logs.push({
                level: "info",
                message: `DCE: removed dead node "${nodeId}" (op=${node.op})`,
            });
        }
        // ── Phase 3: warn about unconsumed graph-input tensors ────────────────────
        // We intentionally do NOT remove graph inputs even if they are unused:
        // they form part of the graph's external interface.
        for (const inputTid of workGraph.inputIds) {
            if (!liveTensors.has(inputTid)) {
                const t = workGraph.tensors.get(inputTid);
                logs.push({
                    level: "warn",
                    message: `DCE: graph input "${inputTid}" (${t?.name ?? "?"}) is unused.`,
                });
            }
        }
        logs.push({
            level: "info",
            message: `DeadCodeEliminationPass complete: ${removedNodes} node(s) removed, ` +
                `${removedTensors} tensor(s) removed.`,
        });
        return removedNodes > 0
            ? { graph: workGraph, changed: true, logs }
            : { graph, changed: false, logs };
    }
}
exports.DeadCodeEliminationPass = DeadCodeEliminationPass;
//# sourceMappingURL=deadCodeEliminationPass.js.map