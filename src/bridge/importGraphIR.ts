// ─────────────────────────────────────────────────────────────────────────────
// bridge/importGraphIR.ts
//
// Bridge: GraphIR → compiler Graph
//
// This module converts an IRPackage (produced by the framework frontend) into
// a Graph instance (consumed by the compiler backend).  It is the only place
// in the codebase that imports both the shared IR types and the compiler Graph.
//
// Mapping strategy
// ────────────────
// The framework frontend uses its own local ID namespace (t0, t1, n0, n1, …).
// The compiler's Graph assigns its own IDs via internal counters when
// addInputTensor() and addNode() are called.
//
// The bridge maintains two explicit mappings:
//
//   frameworkToCompilerTensor: Map<framework TensorId, compiler tensor id>
//   frameworkToCompilerNode:   Map<framework NodeId,   compiler node id>
//
// These maps are returned alongside the Graph so callers can correlate
// framework tensor ids (e.g. parameter ids) with their compiler counterparts.
//
// Only the "forward" graph is imported by default.  Pass `kind: "backward"` to
// import a backward graph instead.
//
// Validation
// ──────────
// The bridge runs validateIRPackage() before importing.  Any structural error
// surfaces immediately as a BridgeError instead of silently producing a
// mis-compiled graph.
// ─────────────────────────────────────────────────────────────────────────────

import { IRPackage, GraphIR }                    from "../shared-ir/schema";
import { TensorId, NodeId }                      from "../shared-ir/ids";
import { validateIRPackage }                     from "../shared-ir/validator";
import { Graph }                                 from "../ir/graph";
import { DType, Shape }                          from "../ir/types";

// ─── Error type ───────────────────────────────────────────────────────────────

export class BridgeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BridgeError";
  }
}

// ─── Result type ─────────────────────────────────────────────────────────────

export interface ImportResult {
  /** The newly constructed compiler Graph. */
  readonly graph: Graph;
  /**
   * Maps each framework-side TensorId to the compiler-assigned tensor id.
   * Use this to cross-reference parameter ids, loss ids, etc.
   */
  readonly tensorIdMap: ReadonlyMap<TensorId, string>;
  /**
   * Maps each framework-side NodeId to the compiler-assigned node id.
   */
  readonly nodeIdMap: ReadonlyMap<NodeId, string>;
}

// ─── Import options ───────────────────────────────────────────────────────────

export interface ImportOptions {
  /**
   * Which graph to import.  Defaults to "forward".
   * If multiple graphs of the requested kind exist, the first is used.
   */
  kind?: "forward" | "backward";
  /**
   * When true, skip the IRPackage structural validation pass.
   * Only use this when you have already validated the package.
   * Defaults to false (always validate).
   */
  skipValidation?: boolean;
  /**
   * Human-readable id to assign to the resulting compiler Graph.
   * Defaults to the GraphIR's own id.
   */
  graphId?: string;
}

// ─── Bridge implementation ───────────────────────────────────────────────────

/**
 * Import a framework IRPackage into a compiler Graph.
 *
 * @throws {BridgeError} on validation failure or missing graph kind.
 */
export function importGraphIR(
  pkg:     IRPackage,
  options: ImportOptions = {},
): ImportResult {
  // ── Validate ────────────────────────────────────────────────────────────
  if (!options.skipValidation) {
    const result = validateIRPackage(pkg);
    if (!result.valid) {
      const lines = result.errors
        .map(e => `  [${e.kind}]: ${e.message}`)
        .join("\n");
      throw new BridgeError(
        `IRPackage validation failed before import:\n${lines}`,
      );
    }
  }

  // ── Find the target GraphIR ──────────────────────────────────────────────
  const kind = options.kind ?? "forward";
  const graphIR = pkg.graphs.find(g => g.kind === kind);
  if (!graphIR) {
    throw new BridgeError(
      `No graph with kind "${kind}" found in IRPackage. ` +
      `Available kinds: ${pkg.graphs.map(g => g.kind).join(", ")}`,
    );
  }

  return importSingleGraphIR(graphIR, options.graphId);
}

/**
 * Import a single GraphIR (without the outer IRPackage wrapper).
 * Useful when working directly with a GraphIR extracted from a package.
 */
export function importSingleGraphIR(
  graphIR: GraphIR,
  graphId?: string,
): ImportResult {
  const graph = new Graph(graphId ?? (graphIR.id as string));
  const tensorIdMap = new Map<TensorId, string>();
  const nodeIdMap   = new Map<NodeId,   string>();

  // ── Step 1: Add all input tensors ────────────────────────────────────────
  // inputIds preserves the declaration order from the framework.
  for (const fwdTid of graphIR.inputIds) {
    const t = graphIR.tensors[fwdTid];
    if (!t) {
      throw new BridgeError(
        `GraphIR input tensor "${fwdTid}" not found in tensors map`,
      );
    }
    const compilerTensor = graph.addInputTensor(
      t.name,
      t.dtype    as DType,
      [...t.shape] as Shape,
    );
    tensorIdMap.set(fwdTid, compilerTensor.id);
  }

  // ── Step 2: Add nodes in topological order ───────────────────────────────
  for (const fwdNid of graphIR.nodeOrder) {
    const node = graphIR.nodes[fwdNid];
    if (!node) {
      throw new BridgeError(
        `GraphIR nodeOrder references node "${fwdNid}" not found in nodes map`,
      );
    }

    // Map framework input tensor ids → compiler tensor ids
    const compilerInputIds: string[] = node.inputs.map(fwdTid => {
      const cid = tensorIdMap.get(fwdTid);
      if (!cid) {
        throw new BridgeError(
          `Node "${fwdNid}" (${node.op}) references input tensor "${fwdTid}" ` +
          `which has not been mapped yet. Ensure nodeOrder is topologically sorted.`,
        );
      }
      return cid;
    });

    // Build output specs from the GraphIR tensor records
    const outputSpecs = node.outputs.map(fwdTid => {
      const t = graphIR.tensors[fwdTid];
      if (!t) {
        throw new BridgeError(
          `Node "${fwdNid}" references output tensor "${fwdTid}" not found in tensors map`,
        );
      }
      return {
        name:  t.name,
        dtype: t.dtype as DType,
        shape: [...t.shape] as Shape,
      };
    });

    const compilerNode = graph.addNode(
      node.op,
      compilerInputIds,
      outputSpecs,
      { ...node.attrs },
    );

    nodeIdMap.set(fwdNid, compilerNode.id);

    // Map framework output tensor ids → compiler tensor ids
    for (let i = 0; i < node.outputs.length; i++) {
      const fwdTid      = node.outputs[i];
      const compilerTid = compilerNode.outputs[i];
      tensorIdMap.set(fwdTid, compilerTid);
    }
  }

  // ── Step 3: Mark graph outputs ───────────────────────────────────────────
  const compilerOutputIds = graphIR.outputIds.map(fwdTid => {
    const cid = tensorIdMap.get(fwdTid);
    if (!cid) {
      throw new BridgeError(
        `Output tensor "${fwdTid}" has no compiler mapping. ` +
        `It may not be produced by any node in the graph.`,
      );
    }
    return cid;
  });

  graph.markOutputs(...compilerOutputIds);

  return { graph, tensorIdMap, nodeIdMap };
}
