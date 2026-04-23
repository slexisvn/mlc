import { GraphIR, IRAttrs, IRDType } from "../ir/schema";
import { TensorId }                  from "../ir/ids";
import { ShapeExpr }                 from "../core/shape";

// ─── Backward pass result ─────────────────────────────────────────────────────

export interface BackwardResult {
  readonly backwardGraph: GraphIR;
  /** forward-param-id → backward-gradient-tensor-id */
  readonly gradMap: ReadonlyMap<TensorId, TensorId>;
}

// ─── Gradient builder context ─────────────────────────────────────────────────

/**
 * Everything a gradient builder needs to compute input gradients for a single
 * forward node.  Passed as a single, extensible context object rather than a
 * long positional argument list so the API can grow (e.g. saved-tensor hooks,
 * output-gradient accumulation strategies) without breaking existing builders.
 */
export interface GradContext {
  /** Forward input shapes — same order as node.inputs. */
  readonly inputShapes:  readonly ShapeExpr[];
  /** Forward input dtypes — same order as node.inputs. */
  readonly inputDtypes:  readonly IRDType[];
  /** Forward output shapes — same order as node.outputs. */
  readonly outputShapes: readonly ShapeExpr[];
  /** Forward output dtypes — same order as node.outputs. */
  readonly outputDtypes: readonly IRDType[];
  /** Attributes from the forward node. */
  readonly attrs:        IRAttrs;

  /**
   * Backward-graph tensor ids for the forward input values (same order as
   * node.inputs).  Use these when a gradient depends on the original input
   * (e.g. ∂L/∂A for mul requires B, ∂L/∂X for matmul requires W).
   */
  readonly inputIds:  readonly string[];

  /**
   * Backward-graph tensor ids for the forward output values (same order as
   * node.outputs).  Available for ops whose gradient formula depends on the
   * forward output (e.g. softmax, sigmoid where σ'(x) = σ(x)·(1−σ(x))).
   */
  readonly outputIds: readonly string[];

  /**
   * Backward-graph tensor ids for the incoming (upstream) gradients — ∂L/∂yᵢ,
   * one per forward output, same order as node.outputs.
   */
  readonly gradIds: readonly string[];

  /**
   * Emit a new op into the backward graph and return its output tensor ids.
   * Grad builders must use this exclusively — never construct graph nodes
   * directly.
   */
  readonly apply: (op: string, inputs: readonly string[], attrs?: IRAttrs) => string[];
}

// ─── Gradient builder function ────────────────────────────────────────────────

/**
 * Compute input gradients for a single forward node.
 *
 * Returns one tensor-id per forward input, in the same order as
 * `node.inputs`.  An empty string in a slot means "no gradient for this
 * input" (e.g. integer indices, non-differentiable operands).
 */
export type GradBuilderFn = (ctx: GradContext) => string[];
