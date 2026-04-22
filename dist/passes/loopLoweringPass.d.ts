import { Graph } from "../ir/graph";
import { Pass, PassResult } from "./pass";
import { LoopModule } from "../ir/loopIR";
export declare class LoopLoweringPass implements Pass {
    readonly name = "LoopLoweringPass";
    private _lastModule;
    /**
     * Return the LoopModule produced by the most recent call to `run()`.
     * Returns null if `run()` has not been called yet.
     */
    getLastModule(): LoopModule | null;
    run(graph: Graph): PassResult;
    /**
     * Build a LoopFunction that computes `outputTid` from graph inputs.
     *
     * The function contains loop nests for all ancestor nodes of `outputTid`
     * in topological order.
     */
    private _buildFunction;
    /**
     * BFS backward from `outputTid` to collect all ancestor node IDs, then
     * return them in the stable topological order given by `graph.nodeOrder`.
     */
    private _findAncestors;
    private _lowerNode;
    /**
     * Generic elementwise lowering.
     *
     * Emits a fully nested loop over the output tensor's shape, with the inner
     * body produced by `computeExpr`.  All input tensors are assumed to be
     * broadcastable to the output shape (no explicit shape-check here).
     */
    private _lowerElementwise;
    /**
     * Lower a 2-D matmul to a canonical 3-level i/j/k loop nest.
     *
     *   for i in [0, M):
     *     for j in [0, N):
     *       out[i, j] = 0.0
     *       for k in [0, K):
     *         out[i, j] += x[i, k] * w[k, j]
     */
    private _lowerMatmul;
    /**
     * Lower a fused linear_relu to a single 3-level i/j/k loop nest.
     *
     * This is the key payoff of operator fusion: the entire computation —
     * matrix-multiply, bias addition, and ReLU activation — executes within
     * one loop nest with no intermediate buffers allocated.
     *
     *   for i in [0, M):
     *     for j in [0, N):
     *       out[i, j] = bias[..., j]           // init accumulator with bias
     *       for k in [0, K):
     *         out[i, j] += x[i, k] * w[k, j]  // accumulate matmul result
     *       out[i, j] = max(0.0, out[i, j])    // apply ReLU in-place
     *
     * `bias` may be 1-D ([N], broadcast) or 2-D ([M, N]).
     *
     * Degrades to matmul lowering if fewer than 3 inputs are present (should
     * not occur after a correctly run FusionPass, but guarded for safety).
     */
    private _lowerLinearRelu;
}
