// ─────────────────────────────────────────────────────────────────────────────
//
// Common Subexpression Elimination (CSE) via value numbering on the graph IR.
//
// Algorithm
// ─────────
// Walk the graph in topological order.  For each node:
//   1. Skip if the op is not pure (impure ops may have observable side-effects
//      that distinguish two otherwise identical invocations).
//   2. Compute a deterministic "value key" for the node:
//        `${op}|${stableAttrs}|${normalizedInputIds}`
//      where `stableAttrs` is a JSON-serialisation of the node's attr map with
//      keys sorted lexicographically, and `normalizedInputIds` has the input
//      tensor ids sorted for commutative ops (add, mul) and ordered otherwise.
//   3. If the value key already exists in the CSE table:
//        a. Wire each consumer of the duplicate node's output tensors to the
//           canonically-produced tensors instead.
//        b. Remove the duplicate node and its output tensors from the graph.
//   4. Otherwise: insert the value key → node id mapping into the CSE table.
//
// Output tensor alignment
// ────────────────────────
// When a node is eliminated, its i-th output tensor is replaced everywhere by
// the i-th output tensor of the canonical node.  Nodes with differing output
// counts are never deduplicated (the value key includes the output count to
// prevent this).  Shapes and dtypes are assumed equal when the key matches —
// this is guaranteed by the op semantics.
//
// Scope
// ─────
// • Pure single-output and multi-output ops are both handled.
// • Layout-transforming ops (transpose, reshape) are pure but excluded by
//   contract (pure=true, but their attrs include layout metadata that makes
//   each application unique in practice — they will match or not based on
//   their attrs key, which is correct behaviour).
// • split / concat are excluded because they are impure in the CSE sense
//   (their output count depends on a run-time attribute).
//   Actually, split/concat are pure=true in the contract but the key already
//   captures attrs fully, so duplicate splits with the same axis/sections are
//   correctly eliminated.
//
// Invariants preserved
// ────────────────────
// • SSA: each non-input tensor retains exactly one producer after elimination.
// • No dangling edges: consumer rewiring is performed before node removal.
// • Graph outputs pointing to eliminated tensors are redirected.
// • validateGraph() passes after this pass.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph }                                          from "../ir/graph";
import { Pass, PassLog, PassResult }                      from "./pass";
import { OpContractRegistry, DEFAULT_CONTRACT_REGISTRY }  from "../ops/opContracts";
import { buildConsumerMap }                               from "../utils/graphUtils";

// ─── Commutative op set ───────────────────────────────────────────────────────

/**
 * Ops whose mathematical result is invariant to input order.
 * Their value key uses lexicographically sorted input ids so that
 *   add(a, b) and add(b, a)
 * produce the same key and are correctly identified as equivalent.
 */
const COMMUTATIVE_OPS = new Set(["add", "mul"]);

// ─── Value key construction ───────────────────────────────────────────────────

/**
 * Produce a stable, collision-resistant string key for `node` that encodes:
 *   - the op name
 *   - the node's attrs (keys sorted, values JSON-serialised)
 *   - the (normalised) input tensor ids
 *   - the number of outputs (so nodes with different output counts never alias)
 *
 * Two nodes with the same key are semantically equivalent and one can be
 * eliminated provided the op is pure.
 */
function makeValueKey(node: { op: string; inputs: readonly string[]; outputs: readonly string[]; attrs: Record<string, unknown> }): string {
  // Sort attrs by key for a deterministic serialisation regardless of
  // insertion order in the original attrs object.
  const sortedAttrs = Object.keys(node.attrs)
    .sort()
    .reduce<Record<string, unknown>>((acc, k) => { acc[k] = node.attrs[k]; return acc; }, {});

  const attrsStr = JSON.stringify(sortedAttrs);

  // Normalise input ids for commutative ops.
  const inputIds = COMMUTATIVE_OPS.has(node.op)
    ? [...node.inputs].sort()
    : [...node.inputs];

  return `${node.op}|${attrsStr}|${inputIds.join(",")}|outputs:${node.outputs.length}`;
}

// ─── Pass ─────────────────────────────────────────────────────────────────────

export class CSEPass implements Pass {
  readonly name = "CSEPass";

  constructor(
    private readonly opRegistry: OpContractRegistry = DEFAULT_CONTRACT_REGISTRY,
  ) {}

  run(graph: Graph): PassResult {
    const logs:      PassLog[] = [];
    const workGraph            = graph.clone();

    let eliminated = 0;

    // value key → id of the canonical node that first produced this computation
    const canonicalNode = new Map<string, string>();

    // snapshot order before any mutation
    const order = [...workGraph.nodeOrder];

    for (const nodeId of order) {
      const node = workGraph.nodes.get(nodeId);
      if (!node) continue; // already removed in this pass (shouldn't happen in CSE)

      // ── Gate: op must be pure ─────────────────────────────────────────────
      if (!this.opRegistry.isPure(node.op)) continue;

      const key = makeValueKey(node);

      const existingCanonicalId = canonicalNode.get(key);

      if (existingCanonicalId === undefined) {
        // First time we see this computation: register as canonical.
        canonicalNode.set(key, nodeId);
        continue;
      }

      // ── Duplicate found ───────────────────────────────────────────────────
      const canonNode = workGraph.nodes.get(existingCanonicalId);
      if (!canonNode) {
        // The canonical node was removed (by DCE or a previous CSE step in a
        // fixpoint iteration).  Re-register this node as the new canonical.
        canonicalNode.set(key, nodeId);
        continue;
      }

      // Safety: output counts must match (always true if the key is correct).
      if (canonNode.outputs.length !== node.outputs.length) {
        logs.push({
          level:   "warn",
          message: `CSE: key collision with mismatched output count for op "${node.op}" — skipping`,
        });
        continue;
      }

      // ── Build an output-tensor remap: duplicate → canonical ───────────────
      // remap[dupTid] = canonTid
      const remap = new Map<string, string>();
      for (let i = 0; i < node.outputs.length; i++) {
        remap.set(node.outputs[i], canonNode.outputs[i]);
      }

      // ── Rewire all consumers of the duplicate's outputs ───────────────────
      const consumers = buildConsumerMap(workGraph);
      for (const [dupTid, canonTid] of remap) {
        const consumerIds = consumers.get(dupTid) ?? [];
        for (const consumerId of consumerIds) {
          const consumer = workGraph.nodes.get(consumerId);
          if (!consumer) continue;
          const newInputs = consumer.inputs.map(tid => tid === dupTid ? canonTid : tid);
          workGraph._replaceNode(consumerId, { ...consumer, inputs: newInputs });
        }
        // Redirect any graph-level outputs that point to the duplicate tensor.
        workGraph._replaceOutputTensor(dupTid, canonTid);
      }

      // ── Remove the duplicate node and its output tensors ──────────────────
      workGraph._removeNode(nodeId);
      for (const dupTid of remap.keys()) {
        workGraph._removeTensor(dupTid);
      }

      logs.push({
        level:   "info",
        message: `CSE: eliminated duplicate node "${nodeId}" (op=${node.op}) — ` +
                 `using canonical "${existingCanonicalId}"`,
      });
      eliminated++;
    }

    logs.push({
      level:   "info",
      message: `CSEPass complete: ${eliminated} duplicate node(s) eliminated.`,
    });

    return eliminated > 0
      ? { graph: workGraph, changed: true,  logs }
      : { graph,            changed: false, logs };
  }
}
