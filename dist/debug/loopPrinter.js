"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// debug/loopPrinter.ts
//
// Console-oriented pretty-printer for LoopModule.
//
// Output style matches printer.ts:
//   • "─".repeat(62) heavy rule for section boundaries.
//   • "·".repeat(62) thin rule between functions.
//   • Two-space indent per loop level, starting at four spaces.
//   • ▸ bullet for top-level items.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.printLoopModule = printLoopModule;
const LINE = "─".repeat(62);
const THIN = "·".repeat(62);
// ── Public API ────────────────────────────────────────────────────────────────
/**
 * Print a LoopModule to stdout.
 *
 * @param module  The module to print.
 * @param title   Optional label shown in the header line.
 */
function printLoopModule(module, title) {
    console.log(`\n${LINE}`);
    if (title)
        console.log(`  ${title}`);
    const fnCount = module.functions.length;
    console.log(`  Loop IR Module : ${module.graphId}` +
        `  (${fnCount} function${fnCount !== 1 ? "s" : ""})`);
    console.log(LINE);
    for (const fn of module.functions) {
        _printFunction(fn);
    }
    if (module.diagnostics.length > 0) {
        console.log(`  ▸ Diagnostics:`);
        for (const d of module.diagnostics) {
            console.log(`      ⚠  ${d}`);
        }
        console.log(THIN);
    }
    console.log(LINE);
}
// ── Private ───────────────────────────────────────────────────────────────────
function _printFunction(fn) {
    const inputs = fn.params.filter(p => p.role === "input");
    const outputs = fn.params.filter(p => p.role === "output");
    const temps = fn.params.filter(p => p.role === "temp");
    const fmtParam = (p) => `${p.name}(${p.shape.map(s => (s === -1 ? "?" : String(s))).join("×")})`;
    const inStr = inputs.length > 0 ? `in: ${inputs.map(fmtParam).join(", ")}` : "in: —";
    const outStr = outputs.length > 0 ? `out: ${outputs.map(fmtParam).join(", ")}` : "out: —";
    const tempStr = temps.length > 0 ? `  [temps: ${temps.map(fmtParam).join(", ")}]` : "";
    console.log(`  ▸ fn ${fn.name}  [${inStr}]  [${outStr}]${tempStr}`);
    if (fn.body.length === 0) {
        console.log(`    (empty body)`);
    }
    else {
        for (const stmt of fn.body) {
            _printStmt(stmt, 4);
        }
    }
    console.log(THIN);
}
/**
 * Recursively print a statement with the given indentation level.
 */
function _printStmt(stmt, indent) {
    const pad = " ".repeat(indent);
    if (stmt.kind === "ForLoop") {
        // Show symbolic hiExpr when the bound is dynamic (edge-tile loops).
        const hiStr = stmt.hi === -1
            ? stmt.hiExpr !== undefined
                ? _fmtExpr(stmt.hiExpr)
                : "?"
            : String(stmt.hi);
        console.log(`${pad}for ${stmt.var.name} in [${stmt.lo}, ${hiStr}):`);
        for (const s of stmt.body) {
            _printStmt(s, indent + 2);
        }
    }
    else {
        // Assign
        const op = stmt.accumulate ? "+=" : "=";
        console.log(`${pad}${_fmtExpr(stmt.target)} ${op} ${_fmtExpr(stmt.value)}`);
    }
}
/**
 * Format a LoopExpr as a compact infix string.
 *
 * Literals are always shown with a decimal point so the reader can
 * distinguish float constants (0.0) from array indices (i0).
 */
function _fmtExpr(e) {
    switch (e.kind) {
        case "LoopVar":
            return e.name;
        case "Literal":
            // Show 0 as 0.0, 1 as 1.0, etc. — clearly float.
            return Number.isInteger(e.value) ? `${e.value}.0` : String(e.value);
        case "MemRef":
            return `${e.buffer}[${e.indices.map(_fmtExpr).join(", ")}]`;
        case "BinOp":
            return `(${_fmtExpr(e.lhs)} ${e.op} ${_fmtExpr(e.rhs)})`;
        case "CallBuiltin":
            return `${e.callee}(${e.args.map(_fmtExpr).join(", ")})`;
    }
}
//# sourceMappingURL=loopPrinter.js.map