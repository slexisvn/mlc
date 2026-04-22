// ─────────────────────────────────────────────────────────────────────────────
// ir/graph.ts
//
// Core IR: Tensor, Node, and Graph.
//
// SSA-like invariants enforced by construction:
//   • Every Tensor has exactly ONE producer (producerNodeId) or is a graph
//     input (producerNodeId === null).
//   • Nodes list input tensor ids and output tensor ids explicitly.
//   • The Graph maintains a stable insertion-order list for determinism.
//
// Mutation methods prefixed with `_` are reserved for the pass system and
// should not be called by user code.
// ─────────────────────────────────────────────────────────────────────────────

import { DType, Shape, Attrs } from "./types";

// ─── Module-level ID counters ─────────────────────────────────────────────────

let _tensorCounter = 0;
let _nodeCounter   = 0;
let _graphCounter  = 0;

/**
 * Reset all ID counters to zero.
 * Useful for test suites that want predictable, repeatable IDs.
 */
export function resetCounters(): void {
  _tensorCounter = 0;
  _nodeCounter   = 0;
  _graphCounter  = 0;
}

// ─── Core IR Interfaces ───────────────────────────────────────────────────────

/**
 * An immutable value flowing through the graph.
 * Each tensor has exactly one producer node (or is a graph input).
 */
export interface Tensor {
  readonly id: string;
  readonly name: string;
  readonly dtype: DType;
  readonly shape: Shape;
  /** null iff this tensor is a graph-level input (no producing node). */
  readonly producerNodeId: string | null;
}

/**
 * An operation node: consumes zero-or-more tensors, produces one-or-more tensors.
 *
 * inputs  = ordered list of consumed tensor ids
 * outputs = ordered list of produced tensor ids
 *
 * Both lists use ids rather than object references to keep the graph serialisable
 * and easy to diff/log.
 */
export interface Node {
  readonly id: string;
  readonly op: string;
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly attrs: Attrs;
}

// ─── Graph ────────────────────────────────────────────────────────────────────

/**
 * A directed acyclic computation graph with SSA-like tensor ownership.
 *
 * Construction API (user-facing):
 *   addInputTensor(...)  — declare a graph input
 *   addNode(...)         — add an operation with its output tensors
 *   markOutputs(...)     — declare which tensors are graph-level outputs
 *
 * Mutation API (pass-internal, prefixed with _):
 *   _replaceNode, _removeNode, _removeTensor, _insertNode, _replaceOutputTensor
 *
 * clone() produces a structurally independent copy so passes can work
 * non-destructively on a snapshot.
 */
export class Graph {
  readonly id: string;

  private _nodeOrder: string[]         = [];
  private _nodes:     Map<string, Node>   = new Map();
  private _tensors:   Map<string, Tensor> = new Map();
  private _inputs:    string[]         = [];
  private _outputs:   string[]         = [];

  constructor(id?: string) {
    this.id = id ?? `graph_${_graphCounter++}`;
  }

  // ─── Public Construction API ──────────────────────────────────────────────

  /**
   * Register a new graph-input tensor (no producer node).
   * These tensors flow into the graph from outside and carry producerNodeId = null.
   */
  addInputTensor(name: string, dtype: DType = "float32", shape: Shape = []): Tensor {
    const t: Tensor = {
      id:             `t${_tensorCounter++}`,
      name,
      dtype,
      shape:          [...shape],
      producerNodeId: null,
    };
    this._tensors.set(t.id, t);
    this._inputs.push(t.id);
    return t;
  }

  /**
   * Add an operation node to the graph.
   *
   * @param op          Op name, e.g. "add", "relu", "matmul".
   * @param inputIds    Tensor ids that this node consumes (must already exist).
   * @param outputSpecs Descriptor for each tensor this node produces.
   * @param attrs       Optional key-value attributes (e.g. { axis: 1 }).
   * @returns The newly created Node.
   */
  addNode(
    op: string,
    inputIds: string[],
    outputSpecs: Array<{ name: string; dtype?: DType; shape?: Shape }>,
    attrs: Attrs = {},
  ): Node {
    // Validate that all input tensors exist
    for (const tid of inputIds) {
      if (!this._tensors.has(tid)) {
        throw new Error(`addNode(${op}): input tensor "${tid}" does not exist`);
      }
    }

    const nodeId = `n${_nodeCounter++}`;

    const outputTensors: Tensor[] = outputSpecs.map(spec => {
      const t: Tensor = {
        id:             `t${_tensorCounter++}`,
        name:           spec.name,
        dtype:          spec.dtype ?? "float32",
        shape:          spec.shape ? [...spec.shape] : [],
        producerNodeId: nodeId,
      };
      this._tensors.set(t.id, t);
      return t;
    });

    const node: Node = {
      id:      nodeId,
      op,
      inputs:  [...inputIds],
      outputs: outputTensors.map(t => t.id),
      attrs:   { ...attrs },
    };

    this._nodes.set(nodeId, node);
    this._nodeOrder.push(nodeId);
    return node;
  }

  /**
   * Declare one or more tensors as graph-level outputs.
   * These are the "observable" results of running the graph.
   */
  markOutputs(...tensorIds: string[]): void {
    for (const id of tensorIds) {
      if (!this._tensors.has(id)) {
        throw new Error(`markOutputs: tensor "${id}" does not exist`);
      }
      if (!this._outputs.includes(id)) this._outputs.push(id);
    }
  }

  // ─── Accessors ─────────────────────────────────────────────────────────────

  getNode(id: string): Node {
    const n = this._nodes.get(id);
    if (!n) throw new Error(`Node not found: "${id}"`);
    return n;
  }

  getTensor(id: string): Tensor {
    const t = this._tensors.get(id);
    if (!t) throw new Error(`Tensor not found: "${id}"`);
    return t;
  }

  /** Nodes in stable insertion order (also valid topological order for a valid graph). */
  get nodeOrder(): readonly string[] { return this._nodeOrder; }

  get nodes(): ReadonlyMap<string, Node>   { return this._nodes;   }
  get tensors(): ReadonlyMap<string, Tensor> { return this._tensors; }
  get inputIds(): readonly string[]        { return this._inputs;  }
  get outputIds(): readonly string[]       { return this._outputs; }

  // ─── Pass-Internal Mutation API (_-prefixed) ──────────────────────────────

  /** Replace a node's record wholesale (e.g. after input-tensor rewiring). */
  _replaceNode(id: string, updated: Node): void {
    if (!this._nodes.has(id)) throw new Error(`_replaceNode: unknown node "${id}"`);
    this._nodes.set(id, updated);
  }

  /** Remove a node and drop it from the order list. */
  _removeNode(id: string): void {
    this._nodes.delete(id);
    this._nodeOrder = this._nodeOrder.filter(n => n !== id);
  }

  /** Remove a tensor record. The caller is responsible for removing dangling edges first. */
  _removeTensor(id: string): void {
    this._tensors.delete(id);
  }

  /**
   * Insert a pre-built node and its associated tensors.
   * @param afterNodeId  When given, inserts immediately after that node in order.
   *                     Falls back to appending if the id is not found.
   */
  _insertNode(node: Node, tensors: Tensor[], afterNodeId?: string): void {
    for (const t of tensors) this._tensors.set(t.id, t);
    this._nodes.set(node.id, node);

    if (afterNodeId !== undefined) {
      const idx = this._nodeOrder.indexOf(afterNodeId);
      if (idx >= 0) {
        this._nodeOrder.splice(idx + 1, 0, node.id);
        return;
      }
    }
    this._nodeOrder.push(node.id);
  }

  /** Swap one graph-output tensor id for another. Used after fusion rewiring. */
  _replaceOutputTensor(oldId: string, newId: string): void {
    this._outputs = this._outputs.map(id => (id === oldId ? newId : id));
  }

  // ─── Cloning ──────────────────────────────────────────────────────────────

  /**
   * Produce a structurally independent shallow clone.
   * Passes should clone the input graph and work on the clone so the original
   * is never mutated — preserving snapshot semantics at each pass boundary.
   */
  clone(): Graph {
    const g = new Graph(this.id + "_opt");

    for (const t of this._tensors.values()) {
      g._tensors.set(t.id, { ...t, shape: [...t.shape] });
    }
    for (const n of this._nodes.values()) {
      g._nodes.set(n.id, {
        ...n,
        inputs:  [...n.inputs],
        outputs: [...n.outputs],
        attrs:   { ...n.attrs },
      });
    }

    g._nodeOrder = [...this._nodeOrder];
    g._inputs    = [...this._inputs];
    g._outputs   = [...this._outputs];
    return g;
  }
}
