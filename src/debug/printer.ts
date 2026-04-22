// ─────────────────────────────────────────────────────────────────────────────
// debug/printer.ts
//
// Console-oriented graph visualisation and execution-plan printer.
//
// printGraph()          — full structured dump of a graph (nodes, tensors,
//                         inputs, outputs).  Used for before/after diffs.
// printExecutionPlan()  — topologically ordered execution steps.
// printDiff()           — side-by-side summary showing node count change.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph } from "../ir/graph";
import { topoSort } from "../utils/toposort";
import { LayoutAnalysisResult } from "../analysis/layoutAnalysis";
import { FusionAnalysisResult } from "../analysis/fusionAnalysis";

const LINE = "─".repeat(62);
const THIN = "·".repeat(62);

// ─── Graph dump ───────────────────────────────────────────────────────────────

/**
 * Print a complete human-readable representation of the graph.
 *
 * @param graph  The graph to print.
 * @param title  Optional header shown above the graph dump.
 */
export function printGraph(graph: Graph, title?: string): void {
  console.log(`\n${LINE}`);
  if (title) console.log(`  ${title}`);
  console.log(`  Graph : ${graph.id}`);
  console.log(`  Nodes : ${graph.nodeOrder.length}   Tensors : ${graph.tensors.size}`);
  console.log(LINE);

  // ── Graph inputs ──────────────────────────────────────────────────────────
  console.log("  ▸ Inputs:");
  if (graph.inputIds.length === 0) {
    console.log("      (none)");
  } else {
    for (const tid of graph.inputIds) {
      const t = graph.getTensor(tid);
      console.log(`      [${tid}] ${_padR(t.name, 18)} dtype=${t.dtype}  shape=[${t.shape.join(",")}]`);
    }
  }

  // ── Graph outputs ─────────────────────────────────────────────────────────
  console.log("  ▸ Outputs:");
  if (graph.outputIds.length === 0) {
    console.log("      (none)");
  } else {
    for (const tid of graph.outputIds) {
      const t = graph.getTensor(tid);
      console.log(`      [${tid}] ${_padR(t.name, 18)} dtype=${t.dtype}  shape=[${t.shape.join(",")}]`);
    }
  }

  // ── Nodes ─────────────────────────────────────────────────────────────────
  console.log(`  ▸ Nodes (${graph.nodeOrder.length}):`);
  for (const nid of graph.nodeOrder) {
    const n = graph.getNode(nid);
    const inStr  = n.inputs .map(tid => _tensorLabel(graph, tid)).join(", ") || "—";
    const outStr = n.outputs.map(tid => _tensorLabel(graph, tid)).join(", ") || "—";
    const attrsStr = _attrsStr(n.attrs);
    console.log(`      [${nid}] ${_padR(n.op, 16)}  in=[${inStr}]`);
    console.log(`      ${" ".repeat(nid.length + 2)}  ${_padR("", 16)}  out=[${outStr}]${attrsStr}`);
  }

  // ── Tensor inventory ──────────────────────────────────────────────────────
  console.log(`  ▸ Tensors (${graph.tensors.size}):`);
  const sortedTensors = [...graph.tensors.values()].sort((a, b) => a.id.localeCompare(b.id));
  for (const t of sortedTensors) {
    const producer = t.producerNodeId ?? "<input>";
    console.log(`      [${t.id}] ${_padR(t.name, 18)} producer=${_padR(producer, 14)} dtype=${t.dtype}`);
  }

  console.log(`${LINE}\n`);
}

// ─── Execution plan ───────────────────────────────────────────────────────────

/**
 * Print a topologically-ordered execution plan for the graph.
 * Each step shows the node id, op name, inputs, and outputs.
 */
export function printExecutionPlan(graph: Graph, title?: string): void {
  const { order, hasCycle } = topoSort(graph);

  console.log(`\n${LINE}`);
  if (title) console.log(`  ${title}`);
  console.log(`  Execution Plan: ${graph.id}`);
  if (hasCycle) {
    console.log("  ⚠ WARNING: graph contains a cycle — plan may be incomplete!");
  }
  console.log(LINE);

  if (order.length === 0) {
    console.log("  (empty graph)");
  } else {
    for (let i = 0; i < order.length; i++) {
      const n      = graph.getNode(order[i]);
      const inStr  = n.inputs .map(tid => _tensorLabel(graph, tid)).join(", ") || "—";
      const outStr = n.outputs.map(tid => _tensorLabel(graph, tid)).join(", ") || "—";
      console.log(`  Step ${String(i + 1).padStart(2)}: [${n.id}] ${_padR(n.op, 16)}`);
      console.log(`           in=[${inStr}]`);
      console.log(`           out=[${outStr}]`);
    }
  }
  console.log(`${LINE}\n`);
}

// ─── Diff summary ─────────────────────────────────────────────────────────────

/**
 * Print a concise before/after summary showing what the optimiser changed.
 */
export function printDiff(before: Graph, after: Graph, passName: string): void {
  const nodeDelta   = before.nodeOrder.length - after.nodeOrder.length;
  const tensorDelta = before.tensors.size    - after.tensors.size;

  console.log(`\n${THIN}`);
  console.log(`  Diff after ${passName}`);
  console.log(THIN);
  console.log(`  Nodes  : ${before.nodeOrder.length} → ${after.nodeOrder.length}  (${_delta(nodeDelta)})`);
  console.log(`  Tensors: ${before.tensors.size}    → ${after.tensors.size}     (${_delta(tensorDelta)})`);

  // Show which ops were removed and which were added.
  const beforeOps = new Set(before.nodeOrder.map(id => `${id}(${before.getNode(id).op})`));
  const afterOps  = new Set(after .nodeOrder.map(id => `${id}(${after .getNode(id).op})`));

  const removed = [...beforeOps].filter(x => !afterOps.has(x));
  const added   = [...afterOps] .filter(x => !beforeOps.has(x));

  if (removed.length) console.log(`  Removed: ${removed.join(", ")}`);
  if (added.length)   console.log(`  Added  : ${added.join(", ")}`);
  console.log(`${THIN}\n`);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function _tensorLabel(graph: Graph, tid: string): string {
  try {
    const t = graph.getTensor(tid);
    return `${t.name}(${tid})`;
  } catch {
    return `?(${tid})`;
  }
}

function _attrsStr(attrs: Record<string, unknown>): string {
  const entries = Object.entries(attrs);
  if (entries.length === 0) return "";
  const body = entries
    .map(([k, v]) => `${k}:${Array.isArray(v) ? `[${(v as unknown[]).join(",")}]` : JSON.stringify(v)}`)
    .join(", ");
  return `  {${body}}`;
}

function _padR(s: string, len: number): string {
  return s.length >= len ? s : s + " ".repeat(len - s.length);
}

function _delta(n: number): string {
  if (n > 0) return `-${n} ✓`;
  if (n < 0) return `+${Math.abs(n)} (increased)`;
  return "0 (unchanged)";
}

// ─── Layout analysis dump ─────────────────────────────────────────────────────

/**
 * Print a summary of layout facts and any detected conflicts / elimination
 * candidates produced by analyzeLayouts().
 */
export function printLayoutAnalysis(result: LayoutAnalysisResult, title?: string): void {
  console.log(`\n${THIN}`);
  if (title) console.log(`  ${title}`);
  console.log(`  Layout Analysis`);
  console.log(THIN);

  console.log(`  ▸ Tensor layout facts (${result.tensorFacts.size}):`);
  for (const [, fact] of result.tensorFacts) {
    const conf = fact.confidence === "certain"  ? "✓" :
                 fact.confidence === "inferred" ? "~" : "?";
    console.log(`      [${fact.tensorId}] ${_padR(fact.layout, 10)} ${conf}  (${fact.source})`);
  }

  if (result.conflicts.length > 0) {
    console.log(`\n  ▸ Layout conflicts (${result.conflicts.length}):`);
    for (const c of result.conflicts) {
      console.log(`      ⚠ ${c.message}`);
    }
  } else {
    console.log(`\n  ▸ Layout conflicts: none`);
  }

  if (result.eliminationCandidates.length > 0) {
    console.log(`\n  ▸ Elimination candidates (${result.eliminationCandidates.length}):`);
    for (const e of result.eliminationCandidates) {
      console.log(`      ✂ ${e.firstNodeId} + ${e.secondNodeId} — ${e.reason}`);
    }
  } else {
    console.log(`\n  ▸ Elimination candidates: none`);
  }

  console.log(`${THIN}\n`);
}

// ─── Fusion analysis dump ─────────────────────────────────────────────────────

/**
 * Print a summary of fusion analysis results including approved candidates
 * and per-node rejection records.
 */
export function printFusionAnalysis(result: FusionAnalysisResult, title?: string): void {
  console.log(`\n${THIN}`);
  if (title) console.log(`  ${title}`);
  console.log(`  Fusion Analysis`);
  console.log(`  Nodes: ${result.stats.totalNodes}  ` +
              `Candidates: ${result.stats.fusibleCandidates}  ` +
              `Rejected: ${result.stats.rejectedNodes}`);
  console.log(THIN);

  if (result.candidates.length > 0) {
    console.log(`  ▸ Approved chains (${result.candidates.length}):`);
    for (const c of result.candidates) {
      const ruleName = c.rule.name ?? c.rule.pattern.join("→");
      console.log(`      ✓ [${c.nodeIds.join(" → ")}]  rule: ${ruleName}`);
    }
  }

  if (result.rejections.length > 0) {
    console.log(`\n  ▸ Rejections (${result.rejections.length}):`);
    for (const r of result.rejections) {
      const rule = r.attemptedRule
        ? ` (rule: ${r.attemptedRule.name ?? r.attemptedRule.pattern.join("→")})`
        : "";
      console.log(`      ✗ ${r.nodeId}(${r.op}) [${r.reason}]${rule}`);
      console.log(`        ${r.detail}`);
    }
  }

  console.log(`${THIN}\n`);
}
