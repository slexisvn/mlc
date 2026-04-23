// ─────────────────────────────────────────────────────────────────────────────
// passes/layoutInsertionPass.ts
//
// Compiler-owned layout insertion pass.
//
// Detects layout conflicts (a layout-sensitive op receives an input whose
// layout does not match any of the op's requiredInputLayouts) and inserts
// "transpose" nodes with perm + fromLayout + toLayout attrs to fix them.
//
// Design
// ──────
// • Uses analyzeLayouts() — the existing analysis pass — to discover conflicts.
// • For each conflict, picks the first requiredLayout for which a known
//   permutation from actualLayout exists in KNOWN_PERMS.
// • If no known permutation covers the conversion (e.g. rank-changing NCHW→NC),
//   the conflict is logged as a warning and skipped.
// • Inserts the transpose node immediately BEFORE the conflicting sensitive op
//   in nodeOrder, maintaining topological validity.
// • Rewires the sensitive node's input slot from the original tensor id to the
//   transposed tensor id.
//
// After this pass runs, LayoutTransformPass can cancel any redundant
// transpose pairs that it created alongside existing ones.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph, Node, Tensor } from "../ir/graph";
import { Pass, PassLog, PassResult } from "./pass";
import { analyzeLayouts } from "../analysis/layoutAnalysis";
import { OpContractRegistry, DEFAULT_CONTRACT_REGISTRY } from "../ops/opContracts";

// ─── Known same-rank permutation table ───────────────────────────────────────
// Keys use the "FROM→TO" format. Only pure axis-permutation conversions are
// listed here. Rank-changing conversions (e.g. NCHW→NC) require a reshape and
// cannot be handled by this pass.
const KNOWN_PERMS: Readonly<Record<string, number[]>> = {
  "NCHW→NHWC": [0, 2, 3, 1],
  "NHWC→NCHW": [0, 3, 1, 2],
  "NCW→NWC":   [0, 2, 1],
  "NWC→NCW":   [0, 2, 1],
};

// ─── Module-level sequence counter for unique inserted node/tensor ids ────────
let _insertionSeq = 0;

// ─────────────────────────────────────────────────────────────────────────────

export class LayoutInsertionPass implements Pass {
  readonly name = "LayoutInsertionPass";

  constructor(
    private readonly opRegistry: OpContractRegistry = DEFAULT_CONTRACT_REGISTRY,
  ) {}

  run(graph: Graph): PassResult {
    const logs: PassLog[] = [];
    const workGraph       = graph.clone();

    const { conflicts } = analyzeLayouts(workGraph, this.opRegistry);

    if (conflicts.length === 0) {
      logs.push({ level: "info", message: "No layout conflicts — no transposes inserted." });
      return { graph, changed: false, logs };
    }

    logs.push({
      level:   "info",
      message: `Found ${conflicts.length} layout conflict(s); attempting to insert transposes.`,
    });

    let inserted = 0;

    for (const conflict of conflicts) {
      // conflict.requiredLayout may be "|"-joined (e.g. "NCHW|NHWC").
      // Pick the first requirement for which a known permutation exists.
      const requiredLayouts = conflict.requiredLayout.split("|");
      let chosenTarget: string | undefined;
      let perm:         number[] | undefined;

      for (const req of requiredLayouts) {
        const key = `${conflict.actualLayout}→${req}`;
        if (KNOWN_PERMS[key]) {
          chosenTarget = req;
          perm         = KNOWN_PERMS[key];
          break;
        }
      }

      if (!perm || !chosenTarget) {
        logs.push({
          level:   "warn",
          message: `No known permutation for "${conflict.actualLayout}→[${conflict.requiredLayout}]" ` +
                   `(node "${conflict.nodeId}" op "${conflict.op}"); skipping.`,
        });
        continue;
      }

      const sensNode    = workGraph.getNode(conflict.nodeId);
      const inputTensor = workGraph.getTensor(conflict.inputTensorId);

      // Locate which input slot carries the conflicting tensor.
      const inputIdx = sensNode.inputs.indexOf(conflict.inputTensorId);
      if (inputIdx < 0) {
        logs.push({
          level:   "warn",
          message: `Conflict tensor "${conflict.inputTensorId}" not found in inputs of ` +
                   `node "${conflict.nodeId}"; skipping.`,
        });
        continue;
      }

      const seq = _insertionSeq++;
      const newTensorId = `ti_${seq}`;
      const newNodeId   = `ni_${seq}`;

      // Build the output tensor for the inserted transpose.
      const transposedTensor: Tensor = {
        id:             newTensorId,
        name:           `${inputTensor.name}_${chosenTarget}`,
        dtype:          inputTensor.dtype,
        shape:          _permuteShape(inputTensor.shape, perm),
        producerNodeId: newNodeId,
      };

      // Build the transpose node.
      const transposeNode: Node = {
        id:      newNodeId,
        op:      "transpose",
        inputs:  [conflict.inputTensorId],
        outputs: [newTensorId],
        attrs:   {
          perm,
          fromLayout: conflict.actualLayout,
          toLayout:   chosenTarget,
        },
      };

      // Rewire the sensitive node to consume the transposed tensor.
      const newInputs = [...sensNode.inputs];
      newInputs[inputIdx] = newTensorId;
      workGraph._replaceNode(conflict.nodeId, { ...sensNode, inputs: newInputs });

      // Insert the transpose immediately before the sensitive op in nodeOrder.
      workGraph._insertNodeBefore(transposeNode, [transposedTensor], conflict.nodeId);

      logs.push({
        level:   "info",
        message: `Inserted ${conflict.actualLayout}→${chosenTarget} transpose ` +
                 `("${newNodeId}") before op "${conflict.op}" ("${conflict.nodeId}").`,
      });

      inserted++;
    }

    if (inserted === 0) {
      return { graph, changed: false, logs };
    }

    logs.push({ level: "info", message: `Layout insertion complete: ${inserted} transpose(s) added.` });
    return { graph: workGraph, changed: true, logs };
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function _permuteShape(shape: readonly number[], perm: number[]): number[] {
  return perm.map(i => shape[i] ?? 0);
}
