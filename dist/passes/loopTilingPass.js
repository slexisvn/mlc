"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// passes/loopTilingPass.ts
//
// Loop-level tiling pass (strip-mining).
//
// Algorithm
// ─────────
// For each LoopFunction in the module:
//   Recursively walk the body.  For each ForLoop:
//     • If eligible (static bound, span ≥ minBound, not a reduction unless
//       tileReductions=true), strip-mine it into outer + inner loops.
//     • Recursively apply the same logic to the resulting inner loop's body
//       (so multi-dimensional tiling is applied with a single pass).
//     • Non-eligible loops are recursed into but not strip-mined.
//
// Tile sizes
// ──────────
// The config provides:
//   defaultTileSize   — fallback tile size when no per-variable override exists.
//   tileSizeByVar     — map from induction-variable name to tile size.
//
// Design notes
// ────────────
// • The pass is non-destructive: it rebuilds ForLoop nodes bottom-up.
// • Strip-mined inner loops inherit `_o`/`_i` variable-name suffixes.
// • For non-divisible bounds the inner loop gets a dynamic `hiExpr` containing
//   `min(T, span − v_o·T)`, printed by loopPrinter as a symbolic expression.
// • The pass reports all tiled dimensions in its logs for observability.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.LoopTilingPass = exports.DEFAULT_TILING_CONFIG = void 0;
const loopIR_1 = require("../ir/loopIR");
const loopAnalysis_1 = require("../analysis/loopAnalysis");
exports.DEFAULT_TILING_CONFIG = {
    defaultTileSize: 32,
    minBound: 64,
    tileReductions: false,
};
// ── Pass ─────────────────────────────────────────────────────────────────────
class LoopTilingPass {
    constructor(config = {}) {
        this.name = "LoopTilingPass";
        this.config = { ...exports.DEFAULT_TILING_CONFIG, ...config };
    }
    run(module) {
        const logs = [];
        let changed = false;
        const newFunctions = [];
        for (const fn of module.functions) {
            const analysis = (0, loopAnalysis_1.analyzeTilingCandidates)(fn, this.config.minBound, this.config.tileReductions);
            for (const rej of analysis.rejections) {
                logs.push({
                    level: "info",
                    message: `  fn=${fn.name} body[${rej.bodyIndex}] not tiled: ${rej.reason}`,
                });
            }
            if (analysis.candidates.length === 0) {
                newFunctions.push(fn);
                continue;
            }
            const newBody = fn.body.map(stmt => stmt.kind === "ForLoop" ? this._tileStmt(stmt, fn.name, logs) : stmt);
            const anyChanged = fn.body.some((s, i) => s !== newBody[i]);
            if (anyChanged)
                changed = true;
            newFunctions.push({ ...fn, body: newBody });
        }
        if (!changed) {
            logs.push({ level: "info", message: "No loop tiling opportunities found." });
            return { module, changed: false, logs };
        }
        const newModule = {
            graphId: module.graphId,
            functions: newFunctions,
            diagnostics: module.diagnostics,
        };
        return { module: newModule, changed: true, logs };
    }
    // ── Recursive tiling ──────────────────────────────────────────────────────
    /**
     * Tile a LoopStmt recursively.
     *
     * If the statement is an eligible ForLoop:
     *   • Strip-mine it into outer + inner loops.
     *   • Recurse into the inner loop's body (so inner dimensions are also tiled).
     * If not eligible:
     *   • Recurse into its body anyway (inner loops may still qualify).
     */
    _tileStmt(stmt, fnName, logs) {
        if (stmt.kind !== "ForLoop")
            return stmt;
        if (this._shouldTile(stmt)) {
            const tileSize = this._getTileSize(stmt);
            const tiled = (0, loopAnalysis_1.stripMine)(stmt, tileSize);
            logs.push({
                level: "info",
                message: `  fn=${fnName}: tiled "${stmt.var.name}" [0,${stmt.hi}) ` +
                    `→ "${tiled.var.name}" × "${tiled.body[0].var.name}" (T=${tileSize})` +
                    (stmt.hi % tileSize !== 0 ? " [edge tile]" : ""),
            });
            // The result of stripMine is: for v_o: [for v_i: original_body_tiled]
            // Recurse into v_i's body to catch inner eligible dimensions.
            const innerLoop = tiled.body[0];
            const newInnerBody = innerLoop.body.map(s => this._tileStmt(s, fnName, logs));
            const anyInnerChanged = innerLoop.body.some((s, i) => s !== newInnerBody[i]);
            if (!anyInnerChanged)
                return tiled;
            const newInnerLoop = innerLoop.hi === -1 && innerLoop.hiExpr !== undefined
                ? (0, loopIR_1.forLoopDyn)(innerLoop.var.name, innerLoop.lo, innerLoop.hiExpr, newInnerBody)
                : (0, loopIR_1.forLoop)(innerLoop.var.name, innerLoop.lo, innerLoop.hi, newInnerBody);
            return (0, loopIR_1.forLoop)(tiled.var.name, tiled.lo, tiled.hi, [newInnerLoop]);
        }
        // Not eligible: recurse into body to catch inner qualifying loops.
        const newBody = stmt.body.map(s => this._tileStmt(s, fnName, logs));
        const anyChanged = stmt.body.some((s, i) => s !== newBody[i]);
        if (!anyChanged)
            return stmt;
        return stmt.hi === -1 && stmt.hiExpr !== undefined
            ? (0, loopIR_1.forLoopDyn)(stmt.var.name, stmt.lo, stmt.hiExpr, newBody)
            : (0, loopIR_1.forLoop)(stmt.var.name, stmt.lo, stmt.hi, newBody);
    }
    // ── Eligibility helpers ───────────────────────────────────────────────────
    _shouldTile(loop) {
        if (loop.hi === -1)
            return false; // dynamic bound
        if (loop.hi - loop.lo < this.config.minBound)
            return false; // too small
        if (!this.config.tileReductions && (0, loopAnalysis_1.isReductionLoop)(loop))
            return false;
        return true;
    }
    _getTileSize(loop) {
        return this.config.tileSizeByVar?.[loop.var.name] ?? this.config.defaultTileSize;
    }
}
exports.LoopTilingPass = LoopTilingPass;
//# sourceMappingURL=loopTilingPass.js.map