import { EagerTensor, Tensor, TensorLike } from "./tensor";
import { IRDType } from "../ir/schema";
import { getActiveBuilder, getActiveParamSink, hasActiveContext } from "../core/context";

function _inferShape(data: any): number[] {
  if (typeof data === "number") return [];
  if (Array.isArray(data)) {
    if (data.length === 0) return [0];
    return [data.length, ..._inferShape(data[0])];
  }
  return [];
}

function _flatten(data: any): number[] {
  if (typeof data === "number") return [data];
  if (Array.isArray(data)) {
    return data.flatMap(_flatten);
  }
  return [];
}

/**
 * Creates a Tensor from raw data.
 * If called inside an ExportSession, directly injects a Constant node into the trace.
 * If called outside, returns an EagerTensor stub.
 */
export function tensor(data: any, options: { dtype?: IRDType } = {}): Tensor {
  const shape = _inferShape(data);
  const flatData = _flatten(data);
  const dtype = options.dtype || "float32";

  if (hasActiveContext()) {
    const gb = getActiveBuilder();
    const sink = getActiveParamSink();
    const name = `const_${Math.random().toString(36).substring(2, 6)}`;
    const t = gb.param(name, dtype, shape);
    sink.push({ tensor: t, data: flatData, isConst: true });
    return t;
  }

  return new EagerTensor(flatData, shape, dtype);
}

function _fill(shape: number[], value: number, dtype: IRDType): Tensor {
  let size = 1;
  for (const dim of shape) size *= dim;
  const flatData = Array(size).fill(value);
  
  if (hasActiveContext()) {
    const gb = getActiveBuilder();
    const sink = getActiveParamSink();
    const name = `const_${Math.random().toString(36).substring(2, 6)}`;
    const t = gb.param(name, dtype, shape);
    sink.push({ tensor: t, data: flatData, isConst: true });
    return t;
  }
  
  return new EagerTensor(flatData, shape, dtype);
}

export function zeros(shape: number[], options: { dtype?: IRDType } = {}): Tensor {
  return _fill(shape, 0, options.dtype || "float32");
}

export function ones(shape: number[], options: { dtype?: IRDType } = {}): Tensor {
  return _fill(shape, 1, options.dtype || "float32");
}
