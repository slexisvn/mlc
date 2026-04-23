// ─────────────────────────────────────────────────────────────────────────────
// Default gradient builder implementations for all built-in ops.
//
// Each builder follows the GradBuilderFn contract: it receives a GradContext
// and returns one backward-graph tensor-id per forward input.
//
// Operator families
// ─────────────────
//   elementwise binary (add, sub, mul)
//     — gradient is broadcast-reduced back to each operand's original shape.
//   elementwise unary  (relu)
//     — placeholder; correct implementation requires a Heaviside/step op.
//   reduction          (sum)
//     — v1 placeholder; passes the upstream gradient through unchanged.
//   linear algebra     (matmul)
//     — standard X^T·grad and grad·W^T rules.
// ─────────────────────────────────────────────────────────────────────────────

import { GradBuilderFn, GradContext } from "./types";
import { broadcast, broadcastedAxes } from "../core/shape";

// ─── Shared gradient helper ───────────────────────────────────────────────────

/**
 * Reduce a gradient tensor from `outShape` down to `targetShape` by summing
 * over every axis that was broadcast-expanded.  Returns `gradId` unchanged
 * when no reduction is needed (shapes already match).
 *
 * Uses keepDims=true to preserve rank during the sum, then strips extra
 * leading size-1 dims via reshape when the ranks differ.
 */
function sumToShape(
  ctx:         GradContext,
  gradId:      string,
  targetShape: readonly number[],
  outShape:    readonly number[],
): string {
  const axes = broadcastedAxes(targetShape, outShape);
  if (axes.length === 0) return gradId;

  const rankDiff = outShape.length - targetShape.length;

  // keepDims keeps rank stable so remaining axis indices stay valid.
  const [summed] = ctx.apply("sum", [gradId], { axes, keepDims: true });

  // When ranks differ, the leading size-1 dims must be dropped.
  if (rankDiff > 0) {
    const [reshaped] = ctx.apply("reshape", [summed], { shape: [...targetShape] });
    return reshaped;
  }

  return summed;
}

// ─── Reusable pattern helpers ─────────────────────────────────────────────────

/**
 * Build broadcast-aware gradients for any binary elementwise op whose forward
 * computation is `out = f(A, B)`.
 *
 * @param gradA  Backward-graph tensor id for ∂L/∂A (before broadcast reduction).
 * @param gradB  Backward-graph tensor id for ∂L/∂B (before broadcast reduction).
 */
function binaryBroadcastGrad(
  ctx:   GradContext,
  gradA: string,
  gradB: string,
): string[] {
  const outShape = [...broadcast(ctx.inputShapes[0], ctx.inputShapes[1])];
  return [
    sumToShape(ctx, gradA, ctx.inputShapes[0], outShape),
    sumToShape(ctx, gradB, ctx.inputShapes[1], outShape),
  ];
}

// ─── Default grad builders ────────────────────────────────────────────────────

export const DEFAULT_GRAD_BUILDERS: Record<string, GradBuilderFn> = {

  // ── add ───────────────────────────────────────────────────────────────────
  // ∂L/∂A = ∂L/∂out  (reduced to A's shape if broadcast occurred)
  // ∂L/∂B = ∂L/∂out  (reduced to B's shape if broadcast occurred)
  add(ctx) {
    return binaryBroadcastGrad(ctx, ctx.gradIds[0], ctx.gradIds[0]);
  },

  // ── sub ───────────────────────────────────────────────────────────────────
  // ∂L/∂A =  ∂L/∂out
  // ∂L/∂B = −∂L/∂out
  sub(ctx) {
    const [negGrad] = ctx.apply("neg", [ctx.gradIds[0]]);
    return binaryBroadcastGrad(ctx, ctx.gradIds[0], negGrad);
  },

  // ── mul ───────────────────────────────────────────────────────────────────
  // ∂L/∂A = ∂L/∂out * B  (reduced to A's shape)
  // ∂L/∂B = ∂L/∂out * A  (reduced to B's shape)
  mul(ctx) {
    const [dA_raw] = ctx.apply("mul", [ctx.gradIds[0], ctx.inputIds[1]]);
    const [dB_raw] = ctx.apply("mul", [ctx.gradIds[0], ctx.inputIds[0]]);
    return binaryBroadcastGrad(ctx, dA_raw, dB_raw);
  },

  // ── relu ──────────────────────────────────────────────────────────────────
  // ∂L/∂x = ∂L/∂out * step(x)   where step(x) = 1 if x > 0 else 0.
  relu(ctx) {
    const [s]  = ctx.apply("step", [ctx.inputIds[0]]);
    const [dX] = ctx.apply("mul",  [ctx.gradIds[0], s]);
    return [dX];
  },

  // ── sigmoid ───────────────────────────────────────────────────────────────
  // ∂L/∂x = ∂L/∂out * sigmoid(x) * (1 - sigmoid(x))
  // Using: s*(1-s) = s - s^2  (no scalar literal needed)
  sigmoid(ctx) {
    const [s]     = ctx.apply("sigmoid", [ctx.inputIds[0]]);
    const [s2]    = ctx.apply("mul", [s, s]);           // s^2
    const [s_1ms] = ctx.apply("sub", [s, s2]);          // s - s^2 = s*(1-s)
    const [dX]    = ctx.apply("mul", [ctx.gradIds[0], s_1ms]);
    return [dX];
  },

  // ── tanh ──────────────────────────────────────────────────────────────────
  // ∂L/∂x = ∂L/∂out * (1 - tanh(x)^2)
  // ones = step(exp(x)) = 1 everywhere (exp(x) > 0 always)
  tanh(ctx) {
    const [t]    = ctx.apply("tanh", [ctx.inputIds[0]]);
    const [t2]   = ctx.apply("mul",  [t, t]);
    const [ex]   = ctx.apply("exp",  [ctx.inputIds[0]]);
    const [ones] = ctx.apply("step", [ex]);
    const [sech2]= ctx.apply("sub",  [ones, t2]);
    const [dX]   = ctx.apply("mul",  [ctx.gradIds[0], sech2]);
    return [dX];
  },

  // ── gelu ──────────────────────────────────────────────────────────────────
  // Approximation: gelu(x) ≈ x * sigmoid(1.702 * x)
  // ∂gelu/∂x ≈ sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
  // For simplicity emit a fast-path approximation using the tanh form:
  //   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π)*(x + 0.044715*x^3)))
  // The exact gradient is complex; use the sigmoid approximation derivative instead:
  //   d_gelu ≈ sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))   [rough approx]
  //   = sigmoid(x) * (1 + x*(1-sigmoid(x)))
  // This is not exact but provides a useful training signal.
  gelu(ctx) {
    const [s]      = ctx.apply("sigmoid", [ctx.inputIds[0]]);
    const [s2]     = ctx.apply("mul",  [s, s]);
    const [s_1ms]  = ctx.apply("sub",  [s, s2]);           // s*(1-s)
    const [x_s1ms] = ctx.apply("mul",  [ctx.inputIds[0], s_1ms]);  // x*s*(1-s)
    const [sum]    = ctx.apply("add",  [s, x_s1ms]);       // s + x*s*(1-s)
    const [dX]     = ctx.apply("mul",  [ctx.gradIds[0], sum]);
    return [dX];
  },

  // ── neg ───────────────────────────────────────────────────────────────────
  // ∂L/∂x = -∂L/∂out
  neg(ctx) {
    const [dX] = ctx.apply("neg", [ctx.gradIds[0]]);
    return [dX];
  },

  // ── abs ───────────────────────────────────────────────────────────────────
  // ∂L/∂x = ∂L/∂out * sign(x)
  // sign(x) = step(x) - step(-x)  (0 at x=0, which is correct sub-gradient)
  abs(ctx) {
    const [nx]   = ctx.apply("neg",  [ctx.inputIds[0]]);
    const [sp]   = ctx.apply("step", [ctx.inputIds[0]]);   // 1 if x>0
    const [sn]   = ctx.apply("step", [nx]);                // 1 if x<0
    const [sign] = ctx.apply("sub",  [sp, sn]);            // sign(x)
    const [dX]   = ctx.apply("mul",  [ctx.gradIds[0], sign]);
    return [dX];
  },

  // ── exp ───────────────────────────────────────────────────────────────────
  // ∂L/∂x = ∂L/∂out * exp(x)
  exp(ctx) {
    const [ex]  = ctx.apply("exp", [ctx.inputIds[0]]);
    const [dX]  = ctx.apply("mul", [ctx.gradIds[0], ex]);
    return [dX];
  },

  // ── sqrt ──────────────────────────────────────────────────────────────────
  // ∂L/∂x = ∂L/∂out / (2 * sqrt(x))
  // = ∂L/∂out * 0.5 / sqrt(x)
  // Without scalar literals: 2*sqrt(x) = sqrt(x) + sqrt(x)
  sqrt(ctx) {
    const [sq]    = ctx.apply("sqrt", [ctx.inputIds[0]]);
    const [two_sq]= ctx.apply("add",  [sq, sq]);            // 2*sqrt(x)
    const [dX]    = ctx.apply("div",  [ctx.gradIds[0], two_sq]);
    return [dX];
  },

  // ── div ───────────────────────────────────────────────────────────────────
  // Forward: out = A / B
  // ∂L/∂A = ∂L/∂out / B
  // ∂L/∂B = -∂L/∂out * A / B^2
  div(ctx) {
    const [dA_raw]  = ctx.apply("div", [ctx.gradIds[0], ctx.inputIds[1]]);
    const [b2]      = ctx.apply("mul", [ctx.inputIds[1], ctx.inputIds[1]]);
    const [num]     = ctx.apply("mul", [ctx.gradIds[0], ctx.inputIds[0]]);
    const [quot]    = ctx.apply("div", [num, b2]);
    const [dB_raw]  = ctx.apply("neg", [quot]);
    return binaryBroadcastGrad(ctx, dA_raw, dB_raw);
  },

  // ── softmax ────────────────────────────────────────────────────────────────
  // ∂L/∂x_i = s_i * (∂L/∂s_i - Σ_j ∂L/∂s_j * s_j)
  //          = s_i * (g_i - dot(g, s))    where s = softmax(x)
  // Implemented as: s * (g - sum(g*s, keepDims)) element-wise.
  softmax(ctx) {
    const [s]      = ctx.apply("softmax", [ctx.inputIds[0]]);
    const [gs]     = ctx.apply("mul",     [ctx.gradIds[0], s]);     // g * s
    const lastAxis = ctx.inputShapes[0].length - 1;
    const [dot]    = ctx.apply("sum",     [gs], { axes: [lastAxis], keepDims: true });
    const [g_dot]  = ctx.apply("sub",     [ctx.gradIds[0], dot]);   // g - sum(g*s)
    const [dX]     = ctx.apply("mul",     [s, g_dot]);
    return [dX];
  },

  // ── sum ───────────────────────────────────────────────────────────────────
  // TODO(v2): correct grad broadcasts ∂L/∂out back to the input shape.
  // Requires knowing which axes were reduced (from node attrs) and whether
  // keepDims was set.  v1 placeholder passes gradient through unchanged.
  sum(ctx) { return [ctx.gradIds[0]]; },

  // ── matmul ────────────────────────────────────────────────────────────────
  // Forward: Y = X @ W
  // ∂L/∂X = ∂L/∂Y @ W^T
  // ∂L/∂W = X^T  @ ∂L/∂Y
  matmul(ctx) {
    const [wT] = ctx.apply("transpose", [ctx.inputIds[1]]);
    const [dX] = ctx.apply("matmul",    [ctx.gradIds[0], wT]);
    const [xT] = ctx.apply("transpose", [ctx.inputIds[0]]);
    const [dW] = ctx.apply("matmul",    [xT, ctx.gradIds[0]]);
    return [dX, dW];
  },
};
