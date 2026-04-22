import { DType, Shape, Attrs, ConstantPayload } from "./types";
/**
 * Reset all ID counters to zero.
 * Useful for test suites that want predictable, repeatable IDs.
 */
export declare function resetCounters(): void;
/**
 * An immutable value flowing through the graph.
 * Each tensor has exactly one producer node (or is a graph input).
 *
 * `constantPayload` is attached by ConstantFoldingPass when it can evaluate
 * a tensor's value at compile time.  Absent on non-constant tensors.
 */
export interface Tensor {
    readonly id: string;
    readonly name: string;
    readonly dtype: DType;
    readonly shape: Shape;
    /** null iff this tensor is a graph-level input (no producing node). */
    readonly producerNodeId: string | null;
    /** Compile-time constant value, if known.  Set by ConstantFoldingPass. */
    readonly constantPayload?: ConstantPayload;
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
export declare class Graph {
    readonly id: string;
    private _nodeOrder;
    private _nodes;
    private _tensors;
    private _inputs;
    private _outputs;
    constructor(id?: string);
    /**
     * Register a new graph-input tensor (no producer node).
     * These tensors flow into the graph from outside and carry producerNodeId = null.
     */
    addInputTensor(name: string, dtype?: DType, shape?: Shape): Tensor;
    /**
     * Add an operation node to the graph.
     *
     * @param op          Op name, e.g. "add", "relu", "matmul".
     * @param inputIds    Tensor ids that this node consumes (must already exist).
     * @param outputSpecs Descriptor for each tensor this node produces.
     * @param attrs       Optional key-value attributes (e.g. { axis: 1 }).
     * @returns The newly created Node.
     */
    addNode(op: string, inputIds: string[], outputSpecs: Array<{
        name: string;
        dtype?: DType;
        shape?: Shape;
    }>, attrs?: Attrs): Node;
    /**
     * Declare one or more tensors as graph-level outputs.
     * These are the "observable" results of running the graph.
     */
    markOutputs(...tensorIds: string[]): void;
    getNode(id: string): Node;
    getTensor(id: string): Tensor;
    /** Nodes in stable insertion order (also valid topological order for a valid graph). */
    get nodeOrder(): readonly string[];
    get nodes(): ReadonlyMap<string, Node>;
    get tensors(): ReadonlyMap<string, Tensor>;
    get inputIds(): readonly string[];
    get outputIds(): readonly string[];
    /** Replace a node's record wholesale (e.g. after input-tensor rewiring). */
    _replaceNode(id: string, updated: Node): void;
    /** Remove a node and drop it from the order list. */
    _removeNode(id: string): void;
    /** Remove a tensor record. The caller is responsible for removing dangling edges first. */
    _removeTensor(id: string): void;
    /**
     * Insert a pre-built node and its associated tensors.
     * @param afterNodeId  When given, inserts immediately after that node in order.
     *                     Falls back to appending if the id is not found.
     */
    _insertNode(node: Node, tensors: Tensor[], afterNodeId?: string): void;
    /** Swap one graph-output tensor id for another. Used after fusion rewiring. */
    _replaceOutputTensor(oldId: string, newId: string): void;
    /**
     * Attach (or overwrite) the compile-time constant payload on a tensor.
     * Used exclusively by ConstantFoldingPass after it evaluates an op.
     */
    _setConstantPayload(tensorId: string, payload: ConstantPayload): void;
    /**
     * Replace a computation node with a "const" source node that has no inputs.
     *
     * After constant folding evaluates a node's output(s), we replace the original
     * compute node with a "const" pseudo-node so that:
     *   1. The SSA invariant is maintained (each output tensor still has exactly
     *      one producer — the new const node).
     *   2. The const node carries no input edges, which lets a subsequent
     *      DeadCodeEliminationPass prune the formerly-feeding upstream nodes.
     *
     * The output tensors keep their existing ids and metadata (including the newly
     * attached `constantPayload`); only their `producerNodeId` is updated.
     *
     * @param nodeId      The id of the node to replace.
     * @param constNodeId A unique id for the replacement const node.
     * @param attrs       Optional extra attrs (e.g. `{ foldedFrom: "add" }`).
     */
    _replaceWithConstNode(nodeId: string, constNodeId: string, attrs?: Attrs): void;
    /**
     * Produce a structurally independent shallow clone.
     * Passes should clone the input graph and work on the clone so the original
     * is never mutated — preserving snapshot semantics at each pass boundary.
     */
    clone(): Graph;
}
