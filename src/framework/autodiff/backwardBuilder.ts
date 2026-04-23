// ─────────────────────────────────────────────────────────────────────────────
// BackwardGraphBuilder — encapsulates the five phases of backward-graph
// construction as distinct, testable methods rather than one monolithic
// function.
//
// Phases
// ──────
//   1. mirrorInputs    — reflect every forward graph input into the bwd graph.
//   2. replayForward   — re-execute forward ops so intermediate tensors are
//                        available as bwd-graph handles for grad formulas.
//   3. seedGradients   — add one bwd-graph input per root (loss) tensor.
//   4. reversePass     — walk nodes in reverse topo order; for each needed
//                        node, invoke its GradBuilderFn, accumulate results.
//   5. collectGradients— mark param grad handles as outputs and seal the graph.
// ─────────────────────────────────────────────────────────────────────────────

import { GraphIR, IRAttrs, IRDType }   from "../ir/schema";
import { TensorId }                    from "../ir/ids";
import { AutodiffError }               from "../core/errors";
import { OpSchemaRegistry }            from "../core/opRegistry";
import { GraphBuilder }                from "../core/graphBuilder";
import { SymbolicTensor }              from "../tensor/tensor";
import { BackwardResult, GradContext } from "./types";
import { GradRegistry }                from "./gradRegistry";
import { AutodiffAnalysis }            from "./backwardAnalysis";

export class BackwardGraphBuilder {
  private readonly _bwdGb: GraphBuilder;

  /** Handles for forward *input* tensors mirrored into the bwd graph. */
  private readonly _fwdIdToBwdHandle    = new Map<TensorId, SymbolicTensor>();
  /** Handles for forward *intermediate* tensors replayed in the bwd graph. */
  private readonly _fwdComputedHandles  = new Map<TensorId, SymbolicTensor>();
  /** Accumulated gradient handle per forward tensor id. */
  private readonly _gradHandles         = new Map<TensorId, SymbolicTensor>();

  constructor(
    private readonly _fwd:          GraphIR,
    private readonly _analysis:     AutodiffAnalysis,
                     opRegistry:    OpSchemaRegistry,
    private readonly _gradRegistry: GradRegistry,
  ) {
    this._bwdGb = new GraphBuilder({ id: "backward", opRegistry });
  }

  // ─── Public entry point ────────────────────────────────────────────────────

  build(rootIds: readonly TensorId[], paramIds: readonly TensorId[]): BackwardResult {
    this._mirrorInputs();
    this._replayForward();
    this._seedGradients(rootIds);
    this._reversePass();
    return this._collectGradients(paramIds);
  }

  // ─── Phase 1 ──────────────────────────────────────────────────────────────

  private _mirrorInputs(): void {
    for (const tid of this._fwd.inputIds) {
      const t = this._fwd.tensors[tid];
      this._fwdIdToBwdHandle.set(
        tid,
        this._bwdGb.input(t.name, t.dtype as IRDType, [...t.shape]),
      );
    }
  }

  // ─── Phase 2 ──────────────────────────────────────────────────────────────

  private _replayForward(): void {
    for (const nid of this._fwd.nodeOrder) {
      const node         = this._fwd.nodes[nid];
      const inputHandles = node.inputs.map(tid => {
        const h = this._fwdIdToBwdHandle.get(tid) ?? this._fwdComputedHandles.get(tid);
        if (!h) throw new AutodiffError(`No backward handle for forward tensor "${tid}"`);
        return h;
      });
      const outHandles = this._bwdGb.applyOp(node.op, inputHandles, node.attrs as IRAttrs);
      for (let i = 0; i < node.outputs.length; i++) {
        this._fwdComputedHandles.set(node.outputs[i], outHandles[i]);
      }
    }
  }

  // ─── Phase 3 ──────────────────────────────────────────────────────────────

  private _seedGradients(rootIds: readonly TensorId[]): void {
    for (const rootId of rootIds) {
      const t = this._fwd.tensors[rootId];
      this._gradHandles.set(
        rootId,
        this._bwdGb.input(`grad_${t.name}`, t.dtype as IRDType, [...t.shape]),
      );
    }
  }

  // ─── Phase 4 ──────────────────────────────────────────────────────────────

  private _reversePass(): void {
    for (const nid of [...this._fwd.nodeOrder].reverse()) {
      const node = this._fwd.nodes[nid];
      if (!this._analysis.neededNodes.has(nid)) continue;

      // Every output of this node must have an accumulated gradient by now.
      const gradTensors = node.outputs.map(outId => {
        const g = this._gradHandles.get(outId);
        if (!g) {
          throw new AutodiffError(
            `No gradient for output "${outId}" of node "${nid}" (${node.op})`,
          );
        }
        return g;
      });

      const builder = this._gradRegistry.get(node.op);
      if (!builder) {
        throw new AutodiffError(
          `Op "${node.op}" has no gradient builder — node "${nid}"`,
        );
      }

      const ctx: GradContext = {
        inputShapes:  node.inputs.map( tid => [...this._fwd.tensors[tid].shape]),
        inputDtypes:  node.inputs.map( tid => this._fwd.tensors[tid].dtype as IRDType),
        outputShapes: node.outputs.map(tid => [...this._fwd.tensors[tid].shape]),
        outputDtypes: node.outputs.map(tid => this._fwd.tensors[tid].dtype as IRDType),
        attrs:        node.attrs as IRAttrs,
        inputIds:     node.inputs.map( tid => this._bwdHandle(tid).id as string),
        outputIds:    node.outputs.map(tid => this._bwdHandle(tid).id as string),
        gradIds:      gradTensors.map( g   => g.id as string),
        apply:        (op, ids, attrs = {}) => {
          const handles = ids.map(id => this._bwdGb.getTensorHandle(id as TensorId));
          return this._bwdGb.applyOp(op, handles, attrs).map(t => t.id as string);
        },
      };

      const inputGradIds = builder(ctx);
      for (let i = 0; i < node.inputs.length; i++) {
        if (inputGradIds[i]) this._accumGrad(node.inputs[i], inputGradIds[i]);
      }
    }
  }

  // ─── Phase 5 ──────────────────────────────────────────────────────────────

  private _collectGradients(paramIds: readonly TensorId[]): BackwardResult {
    const gradMap: Map<TensorId, TensorId> = new Map();
    const gradOutputHandles: SymbolicTensor[] = [];

    for (const pid of paramIds) {
      const g = this._gradHandles.get(pid);
      if (!g) throw new AutodiffError(`No gradient for parameter "${pid}"`);
      gradMap.set(pid, g.id);
      gradOutputHandles.push(g);
    }

    this._bwdGb.markOutputs(...gradOutputHandles);
    return { backwardGraph: this._bwdGb.build("backward"), gradMap };
  }

  // ─── Helpers ──────────────────────────────────────────────────────────────

  private _bwdHandle(fwdTid: TensorId): SymbolicTensor {
    const h = this._fwdIdToBwdHandle.get(fwdTid) ?? this._fwdComputedHandles.get(fwdTid);
    if (!h) {
      throw new AutodiffError(`No backward-side handle for forward tensor "${fwdTid}"`);
    }
    return h;
  }

  private _accumGrad(tid: TensorId, gradId: string): void {
    const g        = this._bwdGb.getTensorHandle(gradId as TensorId);
    const existing = this._gradHandles.get(tid);
    if (!existing) {
      this._gradHandles.set(tid, g);
    } else {
      const [sum] = this._bwdGb.applyOp("add", [existing, g]);
      this._gradHandles.set(tid, sum);
    }
  }
}
