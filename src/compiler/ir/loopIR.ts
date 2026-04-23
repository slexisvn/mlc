// ─────────────────────────────────────────────────────────────────────────────
// Loop IR — explicit loop-nest intermediate representation produced by
// LoopLoweringPass after graph-level optimization.
//
// The Loop IR is intentionally minimal and concrete.  It models only the
// constructs that appear in lowered ML operator kernels: iteration variables,
// memory references, arithmetic, built-in calls, and assignments.
//
// Design notes
// ─────────────
// • Every node carries a `kind` discriminant so callers can exhaustively
//   switch without instanceof checks.
// • Builder helpers (loopVar, memRef, binOp, …) reduce boilerplate in the
//   lowering pass.
// • `nestedLoops` builds a right-to-left wrapped loop nest from flat arrays
//   of variable names and dimension bounds.
// • LoopModule is the top-level container — one LoopFunction per graph output.
// ─────────────────────────────────────────────────────────────────────────────

// ── Expressions ───────────────────────────────────────────────────────────────

/** A reference to the current value of a loop induction variable. */
export interface LoopVar {
  readonly kind: "LoopVar";
  readonly name: string;
}

/** An indexed access to a named buffer, e.g. `out[i, j]`. */
export interface MemRef {
  readonly kind: "MemRef";
  readonly buffer: string;
  readonly indices: readonly LoopExpr[];
}

/** Binary arithmetic expression. */
export interface BinOp {
  readonly kind: "BinOp";
  readonly op: "+" | "-" | "*" | "/";
  readonly lhs: LoopExpr;
  readonly rhs: LoopExpr;
}

/** Call to a named built-in, e.g. `max(0.0, x)`. */
export interface CallBuiltin {
  readonly kind: "CallBuiltin";
  readonly callee: string;
  readonly args: readonly LoopExpr[];
}

/**
 * Numeric constant.
 *
 * When `isIndex` is true the value is an integer used as an array index,
 * loop bound, stride, or similar integer-arithmetic context.  The printer
 * renders it without a decimal point (e.g. `64` instead of `64.0`) so loop
 * kernels are easier to read.  Leave `isIndex` unset (or false) for genuine
 * floating-point scalars such as zero-initializers or relu thresholds.
 */
export interface Literal {
  readonly kind:    "Literal";
  readonly value:   number;
  readonly isIndex?: boolean;
}

export type LoopExpr = LoopVar | MemRef | BinOp | CallBuiltin | Literal;

// ── Statements ────────────────────────────────────────────────────────────────

/**
 * Assignment statement.
 *
 * When `accumulate` is true the statement is rendered as `target += value`
 * (used for reduction loops).  Otherwise it is a plain `target = value`.
 */
export interface Assign {
  readonly kind: "Assign";
  readonly target: MemRef;
  readonly value: LoopExpr;
  readonly accumulate: boolean;
}

/**
 * Counted for-loop.
 *
 * Iterates the induction variable `var.name` from `lo` (inclusive) to `hi`
 * (exclusive) in unit steps.  `hi === -1` signals a dynamic bound; in that
 * case `hiExpr` carries the symbolic upper-bound expression (e.g. the result
 * of a min() for edge-tile handling after loop tiling).
 */
export interface ForLoop {
  readonly kind: "ForLoop";
  readonly var: LoopVar;
  readonly lo: number;
  /** Static upper bound, or -1 when the bound is dynamic (see `hiExpr`). */
  readonly hi: number;
  /** Present iff `hi === -1`. Evaluated in the enclosing scope. */
  readonly hiExpr?: LoopExpr;
  readonly body: readonly LoopStmt[];
}

export type LoopStmt = ForLoop | Assign;

// ── Containers ────────────────────────────────────────────────────────────────

/** A buffer parameter of a LoopFunction. */
export interface LoopParam {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: string;
  /** Semantic role within the function. */
  readonly role: "input" | "output" | "temp";
}

/**
 * A lowered kernel function — maps input buffers to one output buffer.
 *
 * One LoopFunction is emitted per graph output tensor.  The `params` list
 * contains, in order: graph-input buffers, intermediate (temp) buffers, and
 * the output buffer.
 */
export interface LoopFunction {
  readonly name: string;
  readonly params: readonly LoopParam[];
  readonly body: readonly LoopStmt[];
}

/**
 * Top-level container for all lowered kernel functions.
 *
 * `diagnostics` carries human-readable warnings for any nodes that could not
 * be lowered (unsupported ops are skipped with a warning rather than crashing).
 */
export interface LoopModule {
  readonly graphId: string;
  readonly functions: readonly LoopFunction[];
  readonly diagnostics: readonly string[];
}

// ── Builder helpers ───────────────────────────────────────────────────────────

export function loopVar(name: string): LoopVar {
  return { kind: "LoopVar", name };
}

export function memRef(buffer: string, indices: readonly LoopExpr[]): MemRef {
  return { kind: "MemRef", buffer, indices };
}

export function binOp(op: BinOp["op"], lhs: LoopExpr, rhs: LoopExpr): BinOp {
  return { kind: "BinOp", op, lhs, rhs };
}

export function callBuiltin(callee: string, args: readonly LoopExpr[]): CallBuiltin {
  return { kind: "CallBuiltin", callee, args };
}

export function literal(value: number, isIndex?: boolean): Literal {
  return isIndex ? { kind: "Literal", value, isIndex: true } : { kind: "Literal", value };
}

export function assign(target: MemRef, value: LoopExpr, accumulate = false): Assign {
  return { kind: "Assign", target, value, accumulate };
}

export function forLoop(varName: string, lo: number, hi: number, body: readonly LoopStmt[]): ForLoop {
  return { kind: "ForLoop", var: loopVar(varName), lo, hi, body };
}

/**
 * Build a ForLoop with a dynamic upper bound (used for edge-tile loops).
 *
 * Sets `hi = -1` and attaches `hiExpr` so the printer and code generator
 * can emit a runtime min() expression instead of a static constant.
 *
 * Example for a tiled inner loop with non-divisible bound:
 *   `for v_i in [0, min(T, N − v_o*T)):`
 */
export function forLoopDyn(
  varName: string,
  lo: number,
  hiExpr: LoopExpr,
  body: readonly LoopStmt[],
): ForLoop {
  return { kind: "ForLoop", var: loopVar(varName), lo, hi: -1, hiExpr, body };
}

/**
 * Build a right-to-left wrapped nest of ForLoops.
 *
 * `nestedLoops(["i0","i1","i2"], [2,3,4], innerBody)` produces:
 *   for i0 in [0, 2):
 *     for i1 in [0, 3):
 *       for i2 in [0, 4):
 *         <innerBody>
 *
 * Returns `innerBody` unchanged when `varNames` is empty (scalar case).
 */
export function nestedLoops(
  varNames: readonly string[],
  dims: readonly number[],
  innerBody: LoopStmt[],
): LoopStmt[] {
  if (varNames.length === 0) return innerBody;
  let body: LoopStmt[] = innerBody;
  for (let d = varNames.length - 1; d >= 0; d--) {
    body = [forLoop(varNames[d], 0, dims[d], body)];
  }
  return body;
}
