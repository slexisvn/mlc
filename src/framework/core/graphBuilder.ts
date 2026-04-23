// ─────────────────────────────────────────────────────────────────────────────
//
// GraphBuilder — central authority for constructing a computation graph.
// This is an internal class; users should interact via ExportSession.
// ─────────────────────────────────────────────────────────────────────────────

import { SymbolicTensor }                            from "../tensor/tensor";
import { OpSchemaRegistry, defaultOpRegistry }       from "./opRegistry";
import { GraphBuildError }                           from "./errors";
import { IRDType, IRAttrs, GraphIR, TensorIR, NodeIR } from "../ir/schema";
import { TensorId, NodeId, GraphId, asTensorId, asNodeId, asGraphId } from "../ir/ids";
import { ShapeExpr }                                 from "./shape";

// ─────────────────────────────────────────────────────────────────────────────

export interface GraphBuilderOptions {
  id?:           string;
  opRegistry?:   OpSchemaRegistry;
  defaultDtype?: IRDType;
}

export class GraphBuilder {
  private readonly _id:           GraphId;
  private readonly _registry:     OpSchemaRegistry;
  private readonly _defaultDtype: IRDType;

  private _tensorCounter: number = 0;
  private _nodeCounter:   number = 0;

  private readonly _inputIds:  TensorId[] = [];
  private readonly _outputIds: TensorId[] = [];
  private readonly _nodeOrder: NodeId[]   = [];

  private readonly _tensors: Map<TensorId, TensorIR>       = new Map();
  private readonly _nodes:   Map<NodeId,   NodeIR>         = new Map();
  private readonly _handles: Map<TensorId, SymbolicTensor> = new Map();

  private _sealed: boolean = false;

  constructor(options: GraphBuilderOptions = {}) {
    this._id           = asGraphId(options.id ?? "main");
    this._registry     = options.opRegistry  ?? defaultOpRegistry;
    this._defaultDtype = options.defaultDtype ?? "float32";
  }

  private _nextTensorId(): TensorId { return asTensorId(`t${this._tensorCounter++}`); }
  private _nextNodeId():   NodeId   { return asNodeId(`n${this._nodeCounter++}`); }

  private _assertOpen(): void {
    if (this._sealed) {
      throw new GraphBuildError(
        "GraphBuilder has been sealed by build(); no further modifications are allowed",
      );
    }
  }

  private _registerInputTensor(
    name:    string,
    dtype:   IRDType,
    shape:   ShapeExpr,
    isParam: boolean,
  ): SymbolicTensor {
    this._assertOpen();
    const id = this._nextTensorId();
    const tensorIR: TensorIR = { id, name, dtype, shape: [...shape], producerNodeId: null, isParam };
    this._tensors.set(id, tensorIR);
    this._inputIds.push(id);
    const handle = new SymbolicTensor(id, name, dtype, shape, isParam);
    this._handles.set(id, handle);
    return handle;
  }

  input(name: string, dtype: IRDType = this._defaultDtype, shape: ShapeExpr = []): SymbolicTensor {
    return this._registerInputTensor(name, dtype, [...shape], false);
  }

  param(name: string, dtype: IRDType = this._defaultDtype, shape: ShapeExpr = []): SymbolicTensor {
    return this._registerInputTensor(name, dtype, [...shape], true);
  }

  applyOp(
    op:     string,
    inputs: readonly SymbolicTensor[],
    attrs:  IRAttrs = {},
  ): SymbolicTensor[] {
    this._assertOpen();
    const inputShapes = inputs.map(t => t.shape);
    const inputDtypes = inputs.map(t => t.dtype);
    const inferred    = this._registry.infer(op, inputShapes, inputDtypes, attrs);

    const nodeId     = this._nextNodeId();
    const inputIds   = inputs.map(t => t.id);
    const outputIds: TensorId[]       = [];
    const outHandles: SymbolicTensor[] = [];

    for (const out of inferred) {
      const tid = this._nextTensorId();
      const tensorIR: TensorIR = {
        id: tid, name: out.name, dtype: out.dtype,
        shape: [...out.shape], producerNodeId: nodeId,
      };
      this._tensors.set(tid, tensorIR);
      outputIds.push(tid);
      const handle = new SymbolicTensor(tid, out.name, out.dtype, out.shape);
      this._handles.set(tid, handle);
      outHandles.push(handle);
    }

    const nodeIR: NodeIR = { id: nodeId, op, inputs: inputIds, outputs: outputIds, attrs: { ...attrs } };
    this._nodes.set(nodeId, nodeIR);
    this._nodeOrder.push(nodeId);

    return outHandles;
  }

  markOutputs(...tensors: SymbolicTensor[]): void {
    this._assertOpen();
    for (const t of tensors) {
      if (!this._handles.has(t.id)) {
        throw new GraphBuildError(
          `markOutputs: tensor "${t.id}" (${t.name}) does not belong to this GraphBuilder`,
        );
      }
      if (!this._outputIds.includes(t.id)) this._outputIds.push(t.id);
    }
  }

  getTensorHandle(id: TensorId): SymbolicTensor {
    const h = this._handles.get(id);
    if (!h) throw new GraphBuildError(`No tensor with id "${id}" in this GraphBuilder`);
    return h;
  }

  get inputs(): readonly SymbolicTensor[] {
    return this._inputIds.map(id => this._handles.get(id)!);
  }

  get params(): readonly SymbolicTensor[] {
    return this._inputIds.map(id => this._handles.get(id)!).filter(t => t.isParam);
  }

  build(kind: "forward" | "backward" = "forward"): GraphIR {
    this._assertOpen();

    if (this._outputIds.length === 0) {
      if (this._nodeOrder.length === 0) {
        throw new GraphBuildError("Cannot build a graph with no nodes and no outputs");
      }
      const lastNode = this._nodes.get(this._nodeOrder[this._nodeOrder.length - 1])!;
      for (const tid of lastNode.outputs) this._outputIds.push(tid);
    }

    this._sealed = true;

    const tensors: Record<TensorId, TensorIR> = {} as Record<TensorId, TensorIR>;
    for (const [id, t] of this._tensors) tensors[id] = t;

    const nodes: Record<NodeId, NodeIR> = {} as Record<NodeId, NodeIR>;
    for (const [id, n] of this._nodes) nodes[id] = n;

    return {
      id:        this._id,
      kind,
      inputIds:  [...this._inputIds],
      outputIds: [...this._outputIds],
      nodeOrder: [...this._nodeOrder],
      tensors,
      nodes,
    };
  }

  get isSealed(): boolean { return this._sealed; }
}
