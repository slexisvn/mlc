// ─────────────────────────────────────────────────────────────────────────────
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

import { LoopModule, LoopFunction, LoopStmt, ForLoop, forLoop, forLoopDyn } from "../ir/loopIR";
import { PassLog }                                                            from "./pass";
import { LoopPass, LoopPassResult }                                          from "./loopPass";
import {
  isReductionLoop,
  stripMine,
  analyzeTilingCandidates,
} from "../analysis/loopAnalysis";

// ── Configuration ─────────────────────────────────────────────────────────────

export interface TilingConfig {
  /**
   * Tile size used when no per-variable override is present.
   * Default: 32.
   */
  readonly defaultTileSize: number;

  /**
   * Per-induction-variable tile size overrides.
   * Keys are variable names (e.g. "i", "j", "i0", "i1").
   * Values are the desired tile sizes.
   */
  readonly tileSizeByVar?: Readonly<Record<string, number>>;

  /**
   * Minimum static loop span required before tiling is applied.
   * Loops smaller than this are left untouched (avoids unnecessary overhead
   * for tiny dimensions like batch=1 or channels=3).
   * Default: 64.
   */
  readonly minBound: number;

  /**
   * Whether to tile reduction loops (loops where every direct Assign uses
   * accumulate=true, e.g. the matmul k-loop).
   * Default: false — reduction tiling requires careful handling of partial sums
   * and should be opt-in.
   */
  readonly tileReductions: boolean;
}

export const DEFAULT_TILING_CONFIG: TilingConfig = {
  defaultTileSize: 32,
  minBound:        64,
  tileReductions:  false,
};

// ── Pass ─────────────────────────────────────────────────────────────────────

export class LoopTilingPass implements LoopPass {
  readonly name = "LoopTilingPass";

  private readonly config: TilingConfig;

  constructor(config: Partial<TilingConfig> = {}) {
    this.config = { ...DEFAULT_TILING_CONFIG, ...config };
  }

  run(module: LoopModule): LoopPassResult {
    const logs:    PassLog[]           = [];
    let   changed                       = false;
    const newFunctions: LoopFunction[] = [];

    for (const fn of module.functions) {
      const analysis = analyzeTilingCandidates(fn, this.config.minBound, this.config.tileReductions);

      for (const rej of analysis.rejections) {
        logs.push({
          level:   "info",
          message: `  fn=${fn.name} body[${rej.bodyIndex}] not tiled: ${rej.reason}`,
        });
      }

      if (analysis.candidates.length === 0) {
        newFunctions.push(fn);
        continue;
      }

      const newBody = fn.body.map(stmt =>
        stmt.kind === "ForLoop" ? this._tileStmt(stmt, fn.name, logs) : stmt,
      );

      const anyChanged = fn.body.some((s, i) => s !== newBody[i]);
      if (anyChanged) changed = true;

      newFunctions.push({ ...fn, body: newBody });
    }

    if (!changed) {
      logs.push({ level: "info", message: "No loop tiling opportunities found." });
      return { module, changed: false, logs };
    }

    const newModule: LoopModule = {
      graphId:     module.graphId,
      functions:   newFunctions,
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
  private _tileStmt(stmt: LoopStmt, fnName: string, logs: PassLog[]): LoopStmt {
    if (stmt.kind !== "ForLoop") return stmt;

    if (this._shouldTile(stmt)) {
      const tileSize  = this._getTileSize(stmt);
      const tiled     = stripMine(stmt, tileSize);

      logs.push({
        level:   "info",
        message:
          `  fn=${fnName}: tiled "${stmt.var.name}" [0,${stmt.hi}) ` +
          `→ "${tiled.var.name}" × "${(tiled.body[0] as ForLoop).var.name}" (T=${tileSize})` +
          (stmt.hi % tileSize !== 0 ? " [edge tile]" : ""),
      });

      // The result of stripMine is: for v_o: [for v_i: original_body_tiled]
      // Recurse into v_i's body to catch inner eligible dimensions.
      const innerLoop = tiled.body[0] as ForLoop;
      const newInnerBody = innerLoop.body.map(s => this._tileStmt(s, fnName, logs));
      const anyInnerChanged = innerLoop.body.some((s, i) => s !== newInnerBody[i]);

      if (!anyInnerChanged) return tiled;

      const newInnerLoop: ForLoop =
        innerLoop.hi === -1 && innerLoop.hiExpr !== undefined
          ? forLoopDyn(innerLoop.var.name, innerLoop.lo, innerLoop.hiExpr, newInnerBody)
          : forLoop(innerLoop.var.name, innerLoop.lo, innerLoop.hi, newInnerBody);

      return forLoop(tiled.var.name, tiled.lo, tiled.hi, [newInnerLoop]);
    }

    // Not eligible: recurse into body to catch inner qualifying loops.
    const newBody = stmt.body.map(s => this._tileStmt(s, fnName, logs));
    const anyChanged = stmt.body.some((s, i) => s !== newBody[i]);
    if (!anyChanged) return stmt;

    return stmt.hi === -1 && stmt.hiExpr !== undefined
      ? forLoopDyn(stmt.var.name, stmt.lo, stmt.hiExpr, newBody)
      : forLoop(stmt.var.name, stmt.lo, stmt.hi, newBody);
  }

  // ── Eligibility helpers ───────────────────────────────────────────────────

  private _shouldTile(loop: ForLoop): boolean {
    if (loop.hi === -1) return false;                           // dynamic bound
    if (loop.hi - loop.lo < this.config.minBound) return false; // too small
    if (!this.config.tileReductions && isReductionLoop(loop)) return false;
    return true;
  }

  private _getTileSize(loop: ForLoop): number {
    return this.config.tileSizeByVar?.[loop.var.name] ?? this.config.defaultTileSize;
  }
}
