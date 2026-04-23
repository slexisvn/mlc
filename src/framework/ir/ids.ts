// ─────────────────────────────────────────────────────────────────────────────
// shared-ir/ids.ts
//
// Nominal (branded) ID types for the shared IR boundary.
//
// Branding ensures that TensorId, NodeId, and GraphId cannot be accidentally
// mixed with each other or with raw strings at the type level.  The actual
// runtime values are ordinary strings, so there is zero overhead.
//
// Usage:
//   const tid = asTensorId("t0");
//   const nid = asNodeId("n0");
// ─────────────────────────────────────────────────────────────────────────────

/** A string that has been branded as a tensor identifier. */
export type TensorId = string & { readonly __brand: "TensorId" };

/** A string that has been branded as a node identifier. */
export type NodeId = string & { readonly __brand: "NodeId" };

/** A string that has been branded as a graph identifier. */
export type GraphId = string & { readonly __brand: "GraphId" };

/** Cast a raw string to TensorId.  Use only at construction / import sites. */
export function asTensorId(id: string): TensorId {
  return id as TensorId;
}

/** Cast a raw string to NodeId.  Use only at construction / import sites. */
export function asNodeId(id: string): NodeId {
  return id as NodeId;
}

/** Cast a raw string to GraphId.  Use only at construction / import sites. */
export function asGraphId(id: string): GraphId {
  return id as GraphId;
}
