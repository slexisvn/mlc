export type TensorId = string & { readonly __brand: "TensorId" };
export type NodeId = string & { readonly __brand: "NodeId" };
export type GraphId = string & { readonly __brand: "GraphId" };
export function asTensorId(id: string): TensorId {
  return id as TensorId;
}
export function asNodeId(id: string): NodeId {
  return id as NodeId;
}
export function asGraphId(id: string): GraphId {
  return id as GraphId;
}
