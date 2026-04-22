"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// ir/loopIR.ts
//
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.loopVar = loopVar;
exports.memRef = memRef;
exports.binOp = binOp;
exports.callBuiltin = callBuiltin;
exports.literal = literal;
exports.assign = assign;
exports.forLoop = forLoop;
exports.forLoopDyn = forLoopDyn;
exports.nestedLoops = nestedLoops;
// ── Builder helpers ───────────────────────────────────────────────────────────
function loopVar(name) {
    return { kind: "LoopVar", name };
}
function memRef(buffer, indices) {
    return { kind: "MemRef", buffer, indices };
}
function binOp(op, lhs, rhs) {
    return { kind: "BinOp", op, lhs, rhs };
}
function callBuiltin(callee, args) {
    return { kind: "CallBuiltin", callee, args };
}
function literal(value) {
    return { kind: "Literal", value };
}
function assign(target, value, accumulate = false) {
    return { kind: "Assign", target, value, accumulate };
}
function forLoop(varName, lo, hi, body) {
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
function forLoopDyn(varName, lo, hiExpr, body) {
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
function nestedLoops(varNames, dims, innerBody) {
    if (varNames.length === 0)
        return innerBody;
    let body = innerBody;
    for (let d = varNames.length - 1; d >= 0; d--) {
        body = [forLoop(varNames[d], 0, dims[d], body)];
    }
    return body;
}
//# sourceMappingURL=loopIR.js.map