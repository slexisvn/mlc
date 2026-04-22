// ─────────────────────────────────────────────────────────────────────────────
// shared-ir/schema.ts
//
// Plain-data types that form the stable IR boundary between the framework
// frontend and the compiler backend.
//
// Design principles:
//   • All types are plain data (no classes, no methods) for JSON-serializability.
//   • Version fields enable forward/backward compatibility checks.
//   • A GraphIR is self-contained: all tensors and nodes are stored inline.
//   • The forward graph is always present; a backward (grad) graph is optional
//     and lives in a separate GraphIR so compiler passes can run on it
//     independently.
//
// IR version history:
//   "0.1"  — initial version; supports forward and backward graphs.
// ─────────────────────────────────────────────────────────────────────────────

import { TensorId, NodeId, GraphId } from "./ids";

// ─── Primitive aliases ────────────────────────────────────────────────────────

/** Element data type of a tensor. Matches the compiler's DType. */
export type IRDType = "float32" | "float64" | "int32" | "int64" | "bool" | string;

/** Tensor shape.  -1 represents a dynamic / unknown dimension. */
export type IRShape = readonly number[];

/** Generic key-value attributes attached to a node. */
export type IRAttrs = Readonly<Record<string, unknown>>;

// ─── Tensor and Node descriptors ─────────────────────────────────────────────

/**
 * A tensor descriptor in the IR.
 *
 * `producerNodeId` is null for graph inputs (weights, activations fed from
 * outside) and non-null for tensors produced by an operation node.
 *
 * `isParam` marks tensors that carry learnable parameters.  The serializer
 * will emit their data into `IRPackage.parameters`.
 */
export interface TensorIR {
  readonly id:             TensorId;
  /** Human-readable name, e.g. "x", "fc1.weight". */
  readonly name:           string;
  readonly dtype:          IRDType;
  readonly shape:          IRShape;
  /** null for graph-level inputs; id of the producing node otherwise. */
  readonly producerNodeId: NodeId | null;
  /** True when this tensor holds a trainable parameter. */
  readonly isParam?:       boolean;
}

/**
 * An operation node in the IR.
 *
 * `inputs` and `outputs` are ordered; the semantics of each position are
 * defined by the OpSchema in the framework's op registry.
 */
export interface NodeIR {
  readonly id:      NodeId;
  /** Op name, e.g. "matmul", "relu", "add". */
  readonly op:      string;
  /** Ordered input tensor ids. */
  readonly inputs:  readonly TensorId[];
  /** Ordered output tensor ids. */
  readonly outputs: readonly TensorId[];
  readonly attrs:   IRAttrs;
}

// ─── Graph IR ─────────────────────────────────────────────────────────────────

/** Discriminates forward vs. gradient graphs in IRPackage.graphs. */
export type GraphKind = "forward" | "backward";

/**
 * A single computation graph in IR form.
 *
 * `nodeOrder` is a topologically sorted list of node ids; the compiler's
 * passes rely on this invariant to iterate nodes in dependency order.
 *
 * `tensors` and `nodes` use id-keyed plain objects rather than Maps so the
 * structure remains JSON-serializable without any transformation.
 */
export interface GraphIR {
  readonly id:        GraphId;
  readonly kind:      GraphKind;
  /** Tensor ids that are fed from outside the graph (inputs / parameters). */
  readonly inputIds:  readonly TensorId[];
  /** Tensor ids whose values are consumed by the caller after the graph runs. */
  readonly outputIds: readonly TensorId[];
  /** Topologically sorted node execution order. */
  readonly nodeOrder: readonly NodeId[];
  /** All tensors in the graph, keyed by id. */
  readonly tensors:   Readonly<Record<TensorId, TensorIR>>;
  /** All nodes in the graph, keyed by id. */
  readonly nodes:     Readonly<Record<NodeId, NodeIR>>;
}

// ─── Parameter data ───────────────────────────────────────────────────────────

/**
 * Serialized data for a single trainable parameter.
 *
 * `data` is a flat row-major array.  Shape is duplicated here (not just in
 * TensorIR) so parameter files can be read without the full graph schema.
 */
export interface ParameterData {
  readonly tensorId: TensorId;
  readonly name:     string;
  readonly dtype:    IRDType;
  readonly shape:    IRShape;
  /** Flat row-major element values. */
  readonly data:     readonly number[];
}

// ─── Graph signature ─────────────────────────────────────────────────────────

/**
 * Human-readable I/O signature for a graph, useful for documentation and
 * runtime shape-checking without parsing the full GraphIR.
 */
export interface GraphSignature {
  readonly graphId: GraphId;
  readonly inputs:  ReadonlyArray<{ name: string; dtype: IRDType; shape: IRShape }>;
  readonly outputs: ReadonlyArray<{ name: string; dtype: IRDType; shape: IRShape }>;
}

// ─── Top-level package ───────────────────────────────────────────────────────

/**
 * The top-level container exported by the framework and consumed by the bridge.
 *
 * `graphs` always contains at least one "forward" GraphIR.  When autodiff has
 * been run, a second "backward" GraphIR is appended.
 *
 * `parameters` holds the serialized initial values of all trainable tensors.
 * `signatures` provides a lightweight summary of each graph's I/O interface.
 * `metadata` is an open-ended bag for user-supplied annotation.
 */
export interface IRPackage {
  /** IR format version.  This implementation produces "0.1". */
  readonly irVersion:     "0.1";
  /**
   * Op-set version string understood by the consuming backend.
   * The default bridge targets "mini-ts-0.1".
   */
  readonly opsetVersion:  string;
  readonly graphs:        readonly GraphIR[];
  readonly parameters?:   readonly ParameterData[];
  readonly signatures?:   readonly GraphSignature[];
  readonly metadata?:     Readonly<Record<string, unknown>>;
}
