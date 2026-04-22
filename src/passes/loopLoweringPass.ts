// ─────────────────────────────────────────────────────────────────────────────
// passes/loopLoweringPass.ts
//
// Terminal lowering pass: translates an optimized Graph IR into an explicit
// Loop IR (LoopModule) without mutating the graph.
//
// Design contract
// ────────────────
// • Implements the standard Pass interface: run(graph) → PassResult.
// • Returns { graph, changed: false, logs } — graph-preserving by design.
// • The produced LoopModule is retrievable via getLastModule().
// • One LoopFunction is emitted per graph output tensor.
// • Unsupported ops emit a diagnostic warning and are skipped (no crash).
//
// Supported ops
// ──────────────
// Elementwise (any rank):  add, sub, mul, relu, add_relu
// Reduction:               matmul  (2-D; [M,K] × [K,N] → [M,N])
// Fused:                   linear_relu  (matmul + bias + relu in one nest)
//
// Lowering rules
// ──────────────
//   add:         out[…] = a[…] + b[…]
//   sub:         out[…] = a[…] - b[…]
//   mul:         out[…] = a[…] * b[…]
//   relu:        out[…] = max(0.0, x[…])
//   add_relu:    out[…] = max(0.0, a[…] + b[…])            ← fusion payoff
//   matmul:      for i, j: out[i,j]=0; for k: out[i,j]+=x[i,k]*w[k,j]
//   linear_relu: for i, j: out[i,j]=bias; for k: out+=x·w; out=relu(out)
//                                                           ← fusion payoff
// ─────────────────────────────────────────────────────────────────────────────

import { Graph, Node } from "../ir/graph";
import { Pass, PassLog, PassResult } from "./pass";
import {
  LoopModule,
  LoopFunction,
  LoopParam,
  LoopStmt,
  LoopExpr,
  LoopVar,
  MemRef,
  loopVar,
  memRef,
  binOp,
  callBuiltin,
  literal,
  assign,
  forLoop,
  nestedLoops,
} from "../ir/loopIR";

// ── Pass ──────────────────────────────────────────────────────────────────────

export class LoopLoweringPass implements Pass {
  readonly name = "LoopLoweringPass";

  private _lastModule: LoopModule | null = null;

  /**
   * Return the LoopModule produced by the most recent call to `run()`.
   * Returns null if `run()` has not been called yet.
   */
  getLastModule(): LoopModule | null {
    return this._lastModule;
  }

  run(graph: Graph): PassResult {
    const logs: PassLog[]        = [];
    const diagnostics: string[]  = [];
    const functions: LoopFunction[] = [];

    for (const outputTid of graph.outputIds) {
      functions.push(this._buildFunction(graph, outputTid, diagnostics));
    }

    this._lastModule = { graphId: graph.id, functions, diagnostics };

    logs.push({
      level: "info",
      message:
        `Lowered ${graph.outputIds.length} output(s) → ` +
        `${functions.length} LoopFunction(s)`,
    });

    for (const d of diagnostics) {
      logs.push({ level: "warn", message: d });
    }

    // Graph is intentionally NOT mutated — this is a terminal artifact pass.
    return { graph, changed: false, logs };
  }

  // ── Function builder ───────────────────────────────────────────────────────

  /**
   * Build a LoopFunction that computes `outputTid` from graph inputs.
   *
   * The function contains loop nests for all ancestor nodes of `outputTid`
   * in topological order.
   */
  private _buildFunction(
    graph: Graph,
    outputTid: string,
    diagnostics: string[],
  ): LoopFunction {
    const nodeIds = this._findAncestors(graph, outputTid);

    // Collect all tensor IDs that are produced within this node set.
    const producedByChain = new Set<string>();
    for (const nid of nodeIds) {
      for (const tid of graph.getNode(nid).outputs) producedByChain.add(tid);
    }

    const seenTids = new Set<string>();
    const params: LoopParam[] = [];

    // 1. Input params — tensors consumed by the chain but not produced by it.
    for (const nid of nodeIds) {
      for (const tid of graph.getNode(nid).inputs) {
        if (!producedByChain.has(tid) && !seenTids.has(tid)) {
          seenTids.add(tid);
          const t = graph.getTensor(tid);
          params.push({ name: t.name, shape: t.shape, dtype: t.dtype, role: "input" });
        }
      }
    }

    // 2. Temp params — intermediate tensors produced by the chain that are
    //    not the final output (i.e., they flow between unfused nodes).
    for (const nid of nodeIds) {
      for (const tid of graph.getNode(nid).outputs) {
        if (tid !== outputTid && !seenTids.has(tid)) {
          seenTids.add(tid);
          const t = graph.getTensor(tid);
          params.push({ name: t.name, shape: t.shape, dtype: t.dtype, role: "temp" });
        }
      }
    }

    // 3. Output param — always last.
    const outTensor = graph.getTensor(outputTid);
    params.push({
      name:  outTensor.name,
      shape: outTensor.shape,
      dtype: outTensor.dtype,
      role:  "output",
    });

    // 4. Emit loop nests for each node in topological order.
    const body: LoopStmt[] = [];
    for (const nid of nodeIds) {
      const node = graph.getNode(nid);
      const { stmts, diagnostic } = this._lowerNode(node, graph);
      body.push(...stmts);
      if (diagnostic) diagnostics.push(diagnostic);
    }

    // Sanitise the function name — replace non-identifier chars with "_".
    const fnName = `compute_${outTensor.name.replace(/[^a-zA-Z0-9_]/g, "_")}`;
    return { name: fnName, params, body };
  }

  // ── Graph traversal ────────────────────────────────────────────────────────

  /**
   * BFS backward from `outputTid` to collect all ancestor node IDs, then
   * return them in the stable topological order given by `graph.nodeOrder`.
   */
  private _findAncestors(graph: Graph, outputTid: string): string[] {
    const outTensor = graph.getTensor(outputTid);
    if (!outTensor.producerNodeId) return [];   // graph input — no nodes needed

    const visited = new Set<string>();
    const queue: string[] = [outTensor.producerNodeId];

    while (queue.length > 0) {
      const nid = queue.shift()!;
      if (visited.has(nid)) continue;
      visited.add(nid);

      for (const inputTid of graph.getNode(nid).inputs) {
        const t = graph.getTensor(inputTid);
        if (t.producerNodeId) queue.push(t.producerNodeId);
      }
    }

    // Filter graph.nodeOrder (already topo-sorted) to preserve ordering.
    return (graph.nodeOrder as string[]).filter(nid => visited.has(nid));
  }

  // ── Op dispatch ───────────────────────────────────────────────────────────

  private _lowerNode(
    node: Node,
    graph: Graph,
  ): { stmts: LoopStmt[]; diagnostic?: string } {
    switch (node.op) {
      case "add":
        return {
          stmts: this._lowerElementwise(node, graph, ([a, b]) => binOp("+", a, b)),
        };

      case "sub":
        return {
          stmts: this._lowerElementwise(node, graph, ([a, b]) => binOp("-", a, b)),
        };

      case "mul":
        return {
          stmts: this._lowerElementwise(node, graph, ([a, b]) => binOp("*", a, b)),
        };

      case "relu":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("max", [literal(0), x]),
          ),
        };

      case "add_relu":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([a, b]) => callBuiltin("max", [literal(0), binOp("+", a, b)]),
          ),
        };

      case "matmul":
        return { stmts: this._lowerMatmul(node, graph) };

      case "linear_relu":
        return { stmts: this._lowerLinearRelu(node, graph) };

      default:
        return {
          stmts: [],
          diagnostic:
            `Node "${node.id}" (${node.op}): no lowering rule available — skipped.`,
        };
    }
  }

  // ── Elementwise lowering (any rank) ───────────────────────────────────────

  /**
   * Generic elementwise lowering.
   *
   * Emits a fully nested loop over the output tensor's shape, with the inner
   * body produced by `computeExpr`.  All input tensors are assumed to be
   * broadcastable to the output shape (no explicit shape-check here).
   */
  private _lowerElementwise(
    node: Node,
    graph: Graph,
    computeExpr: (inputRefs: MemRef[]) => LoopExpr,
  ): LoopStmt[] {
    const outTid    = node.outputs[0];
    const outTensor = graph.getTensor(outTid);
    const outShape  = outTensor.shape;

    // Build induction variable names i0, i1, … for each dimension.
    const varNames  = outShape.map((_, i) => `i${i}`);
    const vars: LoopVar[] = varNames.map(n => loopVar(n));

    const outRef    = memRef(outTensor.name, vars);
    const inputRefs = node.inputs.map(tid => memRef(graph.getTensor(tid).name, vars));

    const innerBody: LoopStmt[] = [assign(outRef, computeExpr(inputRefs))];
    return nestedLoops(varNames, outShape, innerBody);
  }

  // ── Matmul lowering (2-D) ─────────────────────────────────────────────────

  /**
   * Lower a 2-D matmul to a canonical 3-level i/j/k loop nest.
   *
   *   for i in [0, M):
   *     for j in [0, N):
   *       out[i, j] = 0.0
   *       for k in [0, K):
   *         out[i, j] += x[i, k] * w[k, j]
   */
  private _lowerMatmul(node: Node, graph: Graph): LoopStmt[] {
    const [xTid, wTid] = node.inputs;
    const [outTid]     = node.outputs;

    const xShape = graph.getTensor(xTid).shape;
    const wShape = graph.getTensor(wTid).shape;

    const M = xShape[0] ?? 1;
    const K = xShape[1] ?? 1;
    const N = wShape.length >= 2 ? (wShape[1] ?? 1) : (wShape[0] ?? 1);

    const xName   = graph.getTensor(xTid).name;
    const wName   = graph.getTensor(wTid).name;
    const outName = graph.getTensor(outTid).name;

    const iV = loopVar("i");
    const jV = loopVar("j");
    const kV = loopVar("k");

    const outIJ = memRef(outName, [iV, jV]);
    const xIK   = memRef(xName,   [iV, kV]);
    const wKJ   = memRef(wName,   [kV, jV]);

    const kBody: LoopStmt[] = [
      assign(outIJ, binOp("*", xIK, wKJ), true),  // out[i,j] += x[i,k] * w[k,j]
    ];

    const jBody: LoopStmt[] = [
      assign(outIJ, literal(0)),                   // out[i,j] = 0.0  (init)
      forLoop("k", 0, K, kBody),                   // reduction over k
    ];

    return [forLoop("i", 0, M, [forLoop("j", 0, N, jBody)])];
  }

  // ── linear_relu lowering (fused matmul + bias-add + relu) ────────────────

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
  private _lowerLinearRelu(node: Node, graph: Graph): LoopStmt[] {
    if (node.inputs.length < 3) {
      // Degrade gracefully — treat as plain matmul.
      return this._lowerMatmul(node, graph);
    }

    const [xTid, wTid, biasTid] = node.inputs;
    const [outTid]               = node.outputs;

    const xShape    = graph.getTensor(xTid).shape;
    const wShape    = graph.getTensor(wTid).shape;
    const biasShape = graph.getTensor(biasTid).shape;

    const M = xShape[0] ?? 1;
    const K = xShape[1] ?? 1;
    const N = wShape.length >= 2 ? (wShape[1] ?? 1) : (wShape[0] ?? 1);

    const xName    = graph.getTensor(xTid).name;
    const wName    = graph.getTensor(wTid).name;
    const biasName = graph.getTensor(biasTid).name;
    const outName  = graph.getTensor(outTid).name;

    const iV = loopVar("i");
    const jV = loopVar("j");
    const kV = loopVar("k");

    const outIJ = memRef(outName, [iV, jV]);
    const xIK   = memRef(xName,   [iV, kV]);
    const wKJ   = memRef(wName,   [kV, jV]);

    // Bias may be 1-D (broadcast along i) or 2-D.
    const biasRef: MemRef =
      biasShape.length >= 2
        ? memRef(biasName, [iV, jV])
        : memRef(biasName, [jV]);

    const kBody: LoopStmt[] = [
      assign(outIJ, binOp("*", xIK, wKJ), true),               // out[i,j] += x[i,k]*w[k,j]
    ];

    const jBody: LoopStmt[] = [
      assign(outIJ, biasRef),                                   // out[i,j]  = bias[...]
      forLoop("k", 0, K, kBody),                                // reduction loop
      assign(outIJ, callBuiltin("max", [literal(0), outIJ])),   // relu activation
    ];

    return [forLoop("i", 0, M, [forLoop("j", 0, N, jBody)])];
  }
}
