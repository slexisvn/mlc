// ─────────────────────────────────────────────────────────────────────────────
// ir/validateLoop.ts
//
// Structural validator for LoopModule.
//
// Invariants checked
// ──────────────────
//   1. Bound validity     — every ForLoop has lo >= 0 and (hi > lo or hi === -1).
//   2. Buffer declarations — every MemRef buffer name is declared in fn.params.
//   3. Variable scoping   — every LoopVar reference is bound by an enclosing
//                           ForLoop (no free variables at function scope).
//   4. Write targets      — Assign targets must not reference input-role buffers
//                           (emitted as a warning, not an error, because the
//                           lowering never writes to inputs anyway).
//
// Called by LoopPassManager after each pass when validateAfterEachPass = true.
// ─────────────────────────────────────────────────────────────────────────────

import { LoopModule, LoopFunction, LoopStmt, LoopExpr, LoopParam } from "./loopIR";

// ── Public types ──────────────────────────────────────────────────────────────

export type LoopValidationErrorKind =
  | "InvalidBound"
  | "UndeclaredBuffer"
  | "FreeVariable"
  | "WriteToInput";

export interface LoopValidationError {
  readonly kind:     LoopValidationErrorKind;
  readonly fn:       string;
  readonly message:  string;
}

export interface LoopValidationResult {
  readonly valid:    boolean;
  /** Hard errors that indicate a malformed LoopModule. */
  readonly errors:   readonly LoopValidationError[];
  /** Soft warnings (non-fatal anomalies). */
  readonly warnings: readonly LoopValidationError[];
}

// ── Public API ────────────────────────────────────────────────────────────────

export function validateLoopModule(module: LoopModule): LoopValidationResult {
  const errors:   LoopValidationError[] = [];
  const warnings: LoopValidationError[] = [];

  for (const fn of module.functions) {
    _validateFunction(fn, errors, warnings);
  }

  return { valid: errors.length === 0, errors, warnings };
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function _validateFunction(
  fn:       LoopFunction,
  errors:   LoopValidationError[],
  warnings: LoopValidationError[],
): void {
  const declaredBuffers = new Set(fn.params.map(p => p.name));
  const inputBuffers    = new Set(
    fn.params.filter(p => p.role === "input").map(p => p.name),
  );

  const err = (kind: LoopValidationErrorKind, msg: string): void => {
    errors.push({ kind, fn: fn.name, message: msg });
  };
  const warn = (kind: LoopValidationErrorKind, msg: string): void => {
    warnings.push({ kind, fn: fn.name, message: msg });
  };

  for (const stmt of fn.body) {
    _validateStmt(stmt, declaredBuffers, inputBuffers, new Set(), err, warn);
  }
}

function _validateStmt(
  stmt:       LoopStmt,
  buffers:    Set<string>,
  inputs:     Set<string>,
  boundVars:  Set<string>,
  err:        (k: LoopValidationErrorKind, m: string) => void,
  warn:       (k: LoopValidationErrorKind, m: string) => void,
): void {
  if (stmt.kind === "Assign") {
    // Check target buffer is declared.
    if (!buffers.has(stmt.target.buffer)) {
      err("UndeclaredBuffer", `Assign target "${stmt.target.buffer}" is not a declared parameter`);
    }
    // Warn on writes to input buffers.
    if (inputs.has(stmt.target.buffer)) {
      warn("WriteToInput", `Assign writes to input buffer "${stmt.target.buffer}"`);
    }
    // Validate target indices and value expression.
    for (const idx of stmt.target.indices) {
      _validateExpr(idx, buffers, boundVars, err);
    }
    _validateExpr(stmt.value, buffers, boundVars, err);

  } else {
    // ForLoop
    if (stmt.lo < 0) {
      err("InvalidBound", `ForLoop "${stmt.var.name}": lo=${stmt.lo} is negative`);
    }
    if (stmt.hi !== -1 && stmt.hi <= stmt.lo) {
      err("InvalidBound",
        `ForLoop "${stmt.var.name}": hi=${stmt.hi} is not greater than lo=${stmt.lo}`);
    }
    if (stmt.hi === -1 && stmt.hiExpr === undefined) {
      err("InvalidBound",
        `ForLoop "${stmt.var.name}": hi=-1 but no hiExpr provided`);
    }
    if (stmt.hiExpr) {
      _validateExpr(stmt.hiExpr, buffers, boundVars, err);
    }

    // Add the induction variable to the bound set for the body.
    const inner = new Set(boundVars);
    inner.add(stmt.var.name);

    for (const child of stmt.body) {
      _validateStmt(child, buffers, inputs, inner, err, warn);
    }
  }
}

function _validateExpr(
  expr:      LoopExpr,
  buffers:   Set<string>,
  boundVars: Set<string>,
  err:       (k: LoopValidationErrorKind, m: string) => void,
): void {
  switch (expr.kind) {
    case "LoopVar":
      if (!boundVars.has(expr.name)) {
        err("FreeVariable", `LoopVar "${expr.name}" is not bound by any enclosing ForLoop`);
      }
      break;
    case "MemRef":
      if (!buffers.has(expr.buffer)) {
        err("UndeclaredBuffer", `MemRef "${expr.buffer}" is not a declared parameter`);
      }
      for (const idx of expr.indices) _validateExpr(idx, buffers, boundVars, err);
      break;
    case "BinOp":
      _validateExpr(expr.lhs, buffers, boundVars, err);
      _validateExpr(expr.rhs, buffers, boundVars, err);
      break;
    case "CallBuiltin":
      for (const arg of expr.args) _validateExpr(arg, buffers, boundVars, err);
      break;
    case "Literal":
      break;
  }
}
