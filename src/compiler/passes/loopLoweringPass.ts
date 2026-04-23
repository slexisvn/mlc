// ─────────────────────────────────────────────────────────────────────────────
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
// Reduction:               matmul   (2-D; [M,K] × [K,N] → [M,N])
//                          sum      (any rank, arbitrary axes, keepDims)
// Data:                    transpose  (any-rank permutation)
//                          pool2d     (2-D spatial stride window)
//                          reshape    (arbitrary shape, row-major flat-index)
// Fused:                   linear       (matmul + bias-add)
//                          linear_relu  (matmul + bias + relu in one nest)
//
// Lowering rules
// ──────────────
//   add:         out[…] = a[…] + b[…]
//   sub:         out[…] = a[…] - b[…]
//   mul:         out[…] = a[…] * b[…]
//   relu:        out[…] = max(0.0, x[…])
//   add_relu:    out[…] = max(0.0, a[…] + b[…])            ← fusion payoff
//   matmul:      for i, j: out[i,j]=0; for k: out[i,j]+=x[i,k]*w[k,j]
//   sum:         for outer: out[…]=0; for axes: out[…]+=in[…]  (keepDims aware)
//   transpose:   for i0..iN: out[perm(i0..iN)] = in[i0..iN]
//   pool2d:      for n,c,oi,oj: out=0; for ki,kj: out=max(out, in[n,c,oi*s+ki,oj*s+kj])
//   reshape:     out[i…] = in[j…]  via flat←output→input index decomposition
//   linear:      for i, j: out[i,j]=bias; for k: out+=x·w
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
    // Include the tensor id suffix so names are unique even when multiple
    // graph outputs share the same tensor name (e.g. all named "y").
    const safeName = outTensor.name.replace(/[^a-zA-Z0-9_]/g, "_");
    const safeId   = outputTid.replace(/[^a-zA-Z0-9_]/g, "_");
    const fnName   = `compute_${safeName}_${safeId}`;
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

      case "step":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("step", [x]),
          ),
        };

      case "sigmoid":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("sigmoid", [x]),
          ),
        };

      case "tanh":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("tanh", [x]),
          ),
        };

      case "gelu":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("gelu", [x]),
          ),
        };

      case "exp":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("exp", [x]),
          ),
        };

      case "sqrt":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("sqrt", [x]),
          ),
        };

      case "neg":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("neg", [x]),
          ),
        };

      case "abs":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([x]) => callBuiltin("abs", [x]),
          ),
        };

      case "div":
        return {
          stmts: this._lowerElementwise(
            node, graph,
            ([a, b]) => binOp("/", a, b),
          ),
        };

      case "softmax":
        return { stmts: this._lowerSoftmax(node, graph) };

      case "linear_sigmoid":
        return { stmts: this._lowerLinearAct(node, graph, ([x]) => callBuiltin("sigmoid", [x])) };

      case "linear_tanh":
        return { stmts: this._lowerLinearAct(node, graph, ([x]) => callBuiltin("tanh", [x])) };

      case "linear_gelu":
        return { stmts: this._lowerLinearAct(node, graph, ([x]) => callBuiltin("gelu", [x])) };

      case "matmul":
        return { stmts: this._lowerMatmul(node, graph) };

      case "transpose":
        return { stmts: this._lowerTranspose(node, graph) };

      case "pool2d":
        return { stmts: this._lowerPool2d(node, graph) };

      case "linear":
        return { stmts: this._lowerLinear(node, graph) };

      case "linear_relu":
        return { stmts: this._lowerLinearRelu(node, graph) };

      case "sum":
        return { stmts: this._lowerSum(node, graph) };

      case "reshape":
        return { stmts: this._lowerReshape(node, graph) };

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

  // ── transpose lowering (any-rank permutation) ───────────────────────────

  /**
   * Lower a transpose to a loop nest over the output shape.
   *
   * Given `perm` = [p0, p1, …, pN], output indices are i0…iN and the
   * corresponding input indices are i_{p^{-1}(0)}…i_{p^{-1}(N)} — equivalently
   * we iterate the OUTPUT indices and read from the INPUT using the inverse
   * permutation:
   *
   *   for i0 in [0, out.shape[0]):
   *     for i1 in [0, out.shape[1]):
   *       …
   *       out[i0, i1, …] = in[i_{inv[0]}, i_{inv[1]}, …]
   *
   * where `inv[k] = position of k in perm` (i.e. perm[inv[k]] = k).
   */
  private _lowerTranspose(node: Node, graph: Graph): LoopStmt[] {
    const [xTid]  = node.inputs;
    const [outTid] = node.outputs;

    const outShape = graph.getTensor(outTid).shape;
    const rank     = outShape.length;
    const perm     = (node.attrs["perm"] as number[] | undefined) ??
      Array.from({ length: rank }, (_, i) => rank - 1 - i);

    // Invert the permutation: inv[k] = index of k in perm.
    const inv = new Array<number>(rank);
    for (let i = 0; i < rank; i++) inv[perm[i]] = i;

    // Output iteration variables i0..i_{rank-1}.
    const varNames = outShape.map((_, d) => `i${d}`);
    const vars: LoopVar[] = varNames.map(n => loopVar(n));

    // Input index is vars[inv[0]], vars[inv[1]], …
    const inputIndices = inv.map(d => vars[d]);

    const outRef = memRef(graph.getTensor(outTid).name, vars);
    const inRef  = memRef(graph.getTensor(xTid).name,   inputIndices);

    const innerBody: LoopStmt[] = [assign(outRef, inRef)];
    return nestedLoops(varNames, outShape, innerBody);
  }

  // ── pool2d lowering (2-D spatial max-pool with stride) ─────────────────────

  /**
   * Lower a 2-D max-pool to an explicit 6-level loop nest over an NCHW input.
   *
   * Input layout: [N, C, H, W] (NCHW).  Output: [N, C, oH, oW].
   * oH = floor(H / stride),  oW = floor(W / stride).
   *
   *   for n in [0, N):
   *     for c in [0, C):
   *       for oi in [0, oH):
   *         for oj in [0, oW):
   *           out[n, c, oi, oj] = -Inf      // init with −∞
   *           for ki in [0, K):
   *             for kj in [0, K):
   *               out[n,c,oi,oj] = max(out[n,c,oi,oj], in[n,c,oi*s+ki,oj*s+kj])
   */
  private _lowerPool2d(node: Node, graph: Graph): LoopStmt[] {
    const [xTid]   = node.inputs;
    const [outTid] = node.outputs;

    const inShape  = graph.getTensor(xTid).shape;
    const outShape = graph.getTensor(outTid).shape;

    const K = (node.attrs["kernelSize"] as number | undefined) ?? 2;
    const S = (node.attrs["stride"]     as number | undefined) ?? K;

    const N  = inShape[0] ?? 1;
    const C  = inShape[1] ?? 1;
    const oH = outShape[2] ?? 1;
    const oW = outShape[3] ?? 1;

    const xName   = graph.getTensor(xTid).name;
    const outName = graph.getTensor(outTid).name;

    const nV  = loopVar("n");
    const cV  = loopVar("c");
    const oiV = loopVar("oi");
    const ojV = loopVar("oj");
    const kiV = loopVar("ki");
    const kjV = loopVar("kj");

    const outRef = memRef(outName, [nV, cV, oiV, ojV]);

    // in[n, c, oi*S + ki, oj*S + kj]  (stride * outer index + kernel offset)
    // We represent oi*S + ki as a BinOp literal chain:
    //   binOp("+", binOp("*", literal(S, true), oiV), kiV)
    const inRow = binOp("+", binOp("*", literal(S, true), oiV), kiV);
    const inCol = binOp("+", binOp("*", literal(S, true), ojV), kjV);
    const inRef = memRef(xName, [nV, cV, inRow, inCol]);

    // innermost: out = max(out, in[...])
    const kjBody: LoopStmt[] = [
      assign(outRef, callBuiltin("max", [outRef, inRef])),
    ];
    const kiBody: LoopStmt[] = [forLoop("kj", 0, K, kjBody)];

    const ojBody: LoopStmt[] = [
      assign(outRef, callBuiltin("-Inf", [])),   // out = −∞  (identity for max)
      forLoop("ki", 0, K, kiBody),
    ];

    const oiBody: LoopStmt[] = [forLoop("oj", 0, oW, ojBody)];
    const  cBody: LoopStmt[] = [forLoop("oi", 0, oH, oiBody)];
    const  nBody: LoopStmt[] = [forLoop("c",  0, C,   cBody)];

    return [forLoop("n", 0, N, nBody)];
  }

  // ── linear lowering (fused matmul + bias-add) ────────────────────────────

  /**
   * Lower a fused linear to a single 3-level i/j/k loop nest.
   *
   *   for i in [0, M):
   *     for j in [0, N):
   *       out[i, j] = bias[..., j]           // init accumulator with bias
   *       for k in [0, K):
   *         out[i, j] += x[i, k] * w[k, j]  // accumulate matmul result
   *                                           // (no ReLU — linear only)
   */
  private _lowerLinear(node: Node, graph: Graph): LoopStmt[] {
    if (node.inputs.length < 3) {
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

    const biasRef: MemRef =
      biasShape.length >= 2
        ? memRef(biasName, [iV, jV])
        : memRef(biasName, [jV]);

    const kBody: LoopStmt[] = [
      assign(outIJ, binOp("*", xIK, wKJ), true),   // out[i,j] += x[i,k]*w[k,j]
    ];

    const jBody: LoopStmt[] = [
      assign(outIJ, biasRef),                       // out[i,j]  = bias[...]
      forLoop("k", 0, K, kBody),                    // reduction loop (no relu)
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
  /**
   * Lower a `linear_<act>` fused node (matmul + bias + activation).
   * The activation is provided as a callback from the dispatch table.
   */
  private _lowerLinearAct(
    node: Node,
    graph: Graph,
    act: (refs: MemRef[]) => LoopExpr,
  ): LoopStmt[] {
    if (node.inputs.length < 3) return this._lowerMatmul(node, graph);

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

    const outIJ   = memRef(outName,  [iV, jV]);
    const xIK     = memRef(xName,    [iV, kV]);
    const wKJ     = memRef(wName,    [kV, jV]);
    const biasRef = biasShape.length >= 2
      ? memRef(biasName, [iV, jV])
      : memRef(biasName, [jV]);

    const kBody: LoopStmt[] = [assign(outIJ, binOp("*", xIK, wKJ), true)];
    const jBody: LoopStmt[] = [
      assign(outIJ, biasRef),
      forLoop("k", 0, K, kBody),
      assign(outIJ, act([outIJ])),
    ];
    return [forLoop("i", 0, M, [forLoop("j", 0, N, jBody)])];
  }

  // ── softmax lowering (2-D, along last axis) ───────────────────────────────

  /**
   * Lower softmax to a 2-pass loop over each row:
   *   Pass 1: compute row max for numerical stability.
   *   Pass 2: sum of exp(x - max).
   *   Pass 3: divide each element by the sum.
   *
   * For non-2D input falls back to a diagnostic.
   * For 2-D input [M, N]:
   *   for i in [0, M):
   *     max_val = x[i, 0]
   *     for j in [1, N): max_val = max(max_val, x[i, j])
   *     sum_val = 0
   *     for j in [0, N): out[i, j] = exp(x[i,j] - max_val); sum_val += out[i,j]
   *     for j in [0, N): out[i, j] /= sum_val
   */
  private _lowerSoftmax(node: Node, graph: Graph): LoopStmt[] {
    const [xTid]   = node.inputs;
    const [outTid] = node.outputs;

    const xShape  = graph.getTensor(xTid).shape;
    if (xShape.length !== 2) {
      // Emit identity copy for non-2D (unsupported); leave diagnostic via default path.
      return this._lowerElementwise(node, graph, ([x]) => x);
    }

    const M = xShape[0];
    const N = xShape[1];

    const xName   = graph.getTensor(xTid).name;
    const outName = graph.getTensor(outTid).name;

    const iV = loopVar("i");
    const jV = loopVar("j");

    const xIJ   = memRef(xName,   [iV, jV]);
    const outIJ = memRef(outName, [iV, jV]);

    // We need two scalar temps — encode as 1-D single-element refs.
    // Loop IR doesn't have scalar vars; use single-element dim-0 arrays named
    // _softmax_max and _softmax_sum via 0-indexed memRef.
    // Instead, inline the logic by reusing the output buffer for the exp pass
    // and accumulating sum via a dedicated temp named with loopVar pattern.
    // Simplest correct lowering: emit as three sequential j-loops per row.

    const j0V = loopVar("j");  // reused below

    // j-loop bodies
    const initMaxBody: LoopStmt[]  = [assign(memRef("_smax", [iV]),
      callBuiltin("max", [memRef("_smax", [iV]), xIJ]))];
    const expSumBody: LoopStmt[]   = [
      assign(outIJ, callBuiltin("exp", [binOp("-", xIJ, memRef("_smax", [iV]))])),
      assign(memRef("_ssum", [iV]), binOp("+", memRef("_ssum", [iV]), outIJ), false),
    ];
    const normaliseBody: LoopStmt[] = [
      assign(outIJ, binOp("/", outIJ, memRef("_ssum", [iV]))),
    ];

    void j0V;  // suppress unused warning — j-loop var reused

    const rowBody: LoopStmt[] = [
      // Initialise accumulators
      assign(memRef("_smax", [iV]), memRef(xName, [iV, literal(0, true)])),
      assign(memRef("_ssum", [iV]), literal(0)),
      // Pass 1: find row max
      forLoop("j", 1, N, initMaxBody),
      // Pass 2: exp(x - max) and accumulate sum
      forLoop("j", 0, N, expSumBody),
      // Pass 3: normalise
      forLoop("j", 0, N, normaliseBody),
    ];

    return [forLoop("i", 0, M, rowBody)];
  }

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

  // ── sum lowering (reduction along specified axes) ─────────────────────────

  /**
   * Lower a reduction sum to nested outer + inner loop nests.
   *
   * For each element of the output (outer loops over non-reduced dims):
   *   out[outIdx] = 0.0
   *   for each element of the reduction space (inner loops over axes):
   *     out[outIdx] += in[fullIdx]
   *
   * Supports keepDims (output retains input rank with size-1 at reduced dims)
   * and no-keepDims (reduced dims are absent from output shape entirely).
   */
  private _lowerSum(node: Node, graph: Graph): LoopStmt[] {
    const [xTid]   = node.inputs;
    const [outTid] = node.outputs;

    const xShape   = graph.getTensor(xTid).shape;
    const rank     = xShape.length;
    const xName    = graph.getTensor(xTid).name;
    const outName  = graph.getTensor(outTid).name;

    const axesRaw  = (node.attrs["axes"]     as number[] | undefined) ?? [];
    const keepDims = (node.attrs["keepDims"] as boolean  | undefined) ?? false;

    // Normalise axis indices to allow negative values.
    const axisSet = new Set(axesRaw.map(a => ((a % rank) + rank) % rank));

    // Non-reduced dims → outer loop variables r0, r1, …
    const outerDims: Array<{ inputDim: number; varName: string }> = [];
    for (let d = 0; d < rank; d++) {
      if (!axisSet.has(d)) outerDims.push({ inputDim: d, varName: `r${d}` });
    }

    // Reduced dims → inner (accumulation) loop variables k0, k1, …
    const innerDims: Array<{ inputDim: number; varName: string }> = [];
    for (let d = 0; d < rank; d++) {
      if (axisSet.has(d)) innerDims.push({ inputDim: d, varName: `k${d}` });
    }

    // LoopVar objects keyed by input dimension.
    const outerVarOf = new Map(outerDims.map(({ inputDim, varName }) => [inputDim, loopVar(varName)]));
    const innerVarOf = new Map(innerDims.map(({ inputDim, varName }) => [inputDim, loopVar(varName)]));

    // Output indices.
    //   keepDims=true : same rank; reduced positions fixed at literal(0).
    //   keepDims=false: rank = input rank − |axes|; only outer vars present.
    const outIdx: LoopExpr[] = keepDims
      ? Array.from({ length: rank }, (_, d) =>
          (axisSet.has(d) ? literal(0, true) : outerVarOf.get(d)!) as LoopExpr)
      : outerDims.map(({ inputDim }) => outerVarOf.get(inputDim)! as LoopExpr);

    // Input indices: outer var for non-reduced dims, inner var for reduced.
    const inIdx: LoopExpr[] = Array.from({ length: rank }, (_, d) =>
      (axisSet.has(d) ? innerVarOf.get(d)! : outerVarOf.get(d)!) as LoopExpr,
    );

    const outRef = memRef(outName, outIdx);
    const inRef  = memRef(xName,   inIdx);

    // Build reduction nest innermost-first, then reverse-wrap so the first
    // axis in innerDims becomes the outermost reduction loop.
    let reductionNest: LoopStmt[] = [assign(outRef, inRef, true /* += */)];
    for (let i = innerDims.length - 1; i >= 0; i--) {
      const { inputDim, varName } = innerDims[i];
      reductionNest = [forLoop(varName, 0, xShape[inputDim], reductionNest)];
    }

    // The accumulator init and reduction loops form the body of the outer nest.
    const outerBody: LoopStmt[] = [assign(outRef, literal(0)), ...reductionNest];

    return nestedLoops(
      outerDims.map(({ varName }) => varName),
      outerDims.map(({ inputDim }) => xShape[inputDim]),
      outerBody,
    );
  }

  // ── reshape lowering (row-major flat-index encoding/decoding) ─────────────

  /**
   * Lower a reshape to an element-wise copy with explicit index mapping.
   *
   * Iterates the output shape with nested loops.  For each output position
   * the flat (linear) index is computed from the output multi-index using the
   * output row-major strides, then decomposed into the input multi-index:
   *
   *   flat    = i0·S0_out + i1·S1_out + … + i_{R-1}
   *   j_d     = floor(flat / S_d_in) % xShape[d]
   *   out[i…] = in[j0, j1, …]
   *
   * `mod` is emitted as callBuiltin("mod", …); `/` as BinOp("/", …) (floor
   * division for integer indices).
   */
  private _lowerReshape(node: Node, graph: Graph): LoopStmt[] {
    const [xTid]   = node.inputs;
    const [outTid] = node.outputs;

    const xShape   = graph.getTensor(xTid).shape;
    const outShape = graph.getTensor(outTid).shape;
    const xName    = graph.getTensor(xTid).name;
    const outName  = graph.getTensor(outTid).name;

    const outRank = outShape.length;
    const inRank  = xShape.length;

    const outStrides = _rowMajorStrides(outShape);
    const inStrides  = _rowMajorStrides(xShape);

    // Output loop variables i0…i_{outRank-1}.
    const outVarNames = outShape.map((_, i) => `i${i}`);
    const outVars     = outVarNames.map(n => loopVar(n));

    // Flat index: i0*S0 + i1*S1 + … + i_{R-1}  (left-folded sum of terms).
    let flatExpr: LoopExpr = literal(0);
    for (let d = 0; d < outRank; d++) {
      const term: LoopExpr = outStrides[d] === 1
        ? outVars[d]
        : binOp("*", outVars[d], literal(outStrides[d], true));
      flatExpr = d === 0 ? term : binOp("+", flatExpr, term);
    }

    // Input indices: j_d = floor(flat / inStrides[d]) % xShape[d].
    const inIdx: LoopExpr[] = xShape.map((dim, d) => {
      const divided: LoopExpr = inStrides[d] === 1
        ? flatExpr
        : binOp("/", flatExpr, literal(inStrides[d], true));
      return callBuiltin("mod", [divided, literal(dim, true)]);

    });

    const outRef = memRef(outName, outVars);
    const inRef  = memRef(xName,   inIdx);

    return nestedLoops(outVarNames, outShape, [assign(outRef, inRef)]);
  }
}

// ── Module-level helper ───────────────────────────────────────────────────────

/**
 * Compute row-major (C-order) strides for `shape`.
 * stride[d] = product of shape[d+1 .. rank-1]; last stride is always 1.
 */
function _rowMajorStrides(shape: readonly number[]): number[] {
  const s = new Array<number>(shape.length).fill(1);
  for (let d = shape.length - 2; d >= 0; d--) s[d] = s[d + 1] * shape[d + 1];
  return s;
}
