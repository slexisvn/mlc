"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// analysis/loopAnalysis.ts
//
// Loop IR analysis utilities consumed by LoopFusionPass and LoopTilingPass.
//
// Capabilities
// ────────────
// • Perfect-nest detection and extraction (PerfectNestInfo).
// • Iteration-space compatibility checking.
// • Buffer read/write summarization over arbitrary statement trees.
// • Reduction-loop detection.
// • Recursive expression and statement substitution (loop-variable renaming,
//   tiled-index rewriting).
// • Loop-nest reconstruction from a flat level list.
// • Strip-mining: split one ForLoop dimension into outer+inner tile loops.
// • Fusion-candidate discovery (adjacent perfect nests in a LoopFunction body).
// • Tiling-candidate discovery (eligible ForLoops in a LoopFunction body).
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.extractPerfectNest = extractPerfectNest;
exports.isPerfectNest = isPerfectNest;
exports.isSameIterSpace = isSameIterSpace;
exports.collectExprBuffers = collectExprBuffers;
exports.collectReads = collectReads;
exports.collectWrites = collectWrites;
exports.isReductionLoop = isReductionLoop;
exports.substituteExpr = substituteExpr;
exports.substituteStmt = substituteStmt;
exports.rebuildNest = rebuildNest;
exports.stripMine = stripMine;
exports.analyzeFusionCandidates = analyzeFusionCandidates;
exports.analyzeTilingCandidates = analyzeTilingCandidates;
const loopIR_1 = require("../ir/loopIR");
/**
 * If `loop` is a perfect nest, return its descriptor.  Otherwise return null.
 *
 * A perfect nest is defined as: every intermediate nesting level has exactly
 * one body statement (a ForLoop), and the innermost level has one or more
 * Assign statements and no ForLoops.
 */
function extractPerfectNest(loop) {
    const levels = [];
    let cur = loop;
    for (;;) {
        const body = cur.body;
        // Innermost level: all statements must be Assigns.
        if (body.length > 0 && body.every(s => s.kind === "Assign")) {
            levels.push({ name: cur.var.name, lo: cur.lo, hi: cur.hi, hiExpr: cur.hiExpr });
            return { levels, innerBody: body };
        }
        // Intermediate level: exactly one ForLoop, no Assigns.
        if (body.length === 1 && body[0].kind === "ForLoop") {
            levels.push({ name: cur.var.name, lo: cur.lo, hi: cur.hi, hiExpr: cur.hiExpr });
            cur = body[0];
            continue;
        }
        return null; // mixed body — not a perfect nest
    }
}
/** True iff `stmt` is a ForLoop forming a perfect nest. */
function isPerfectNest(stmt) {
    return stmt.kind === "ForLoop" && extractPerfectNest(stmt) !== null;
}
// ── Iteration-space compatibility ─────────────────────────────────────────────
/**
 * Two perfect nests are compatible for fusion when they have the same number
 * of dimensions, the same static bounds at every level, and no dynamic bounds
 * (hi !== -1 at every level).
 */
function isSameIterSpace(a, b) {
    if (a.levels.length !== b.levels.length)
        return false;
    return a.levels.every((la, i) => {
        const lb = b.levels[i];
        return la.hi !== -1 && lb.hi !== -1 && la.lo === lb.lo && la.hi === lb.hi;
    });
}
// ── Buffer access summary ─────────────────────────────────────────────────────
/** Collect all buffer names referenced in an expression tree (read positions). */
function collectExprBuffers(expr, out = new Set()) {
    switch (expr.kind) {
        case "MemRef":
            out.add(expr.buffer);
            for (const idx of expr.indices)
                collectExprBuffers(idx, out);
            break;
        case "BinOp":
            collectExprBuffers(expr.lhs, out);
            collectExprBuffers(expr.rhs, out);
            break;
        case "CallBuiltin":
            for (const arg of expr.args)
                collectExprBuffers(arg, out);
            break;
        case "LoopVar":
        case "Literal":
            break;
    }
    return out;
}
/**
 * Collect all buffer names that appear in read position within `stmts`.
 * Includes MemRefs in Assign value expressions and in index expressions of
 * Assign targets, plus hiExpr references in ForLoop headers.
 */
function collectReads(stmts, out = new Set()) {
    for (const stmt of stmts) {
        if (stmt.kind === "Assign") {
            collectExprBuffers(stmt.value, out);
            for (const idx of stmt.target.indices)
                collectExprBuffers(idx, out);
        }
        else {
            if (stmt.hiExpr)
                collectExprBuffers(stmt.hiExpr, out);
            collectReads(stmt.body, out);
        }
    }
    return out;
}
/**
 * Collect all buffer names that appear as Assign target buffers in `stmts`
 * (i.e., buffers that are written).
 */
function collectWrites(stmts, out = new Set()) {
    for (const stmt of stmts) {
        if (stmt.kind === "Assign") {
            out.add(stmt.target.buffer);
        }
        else {
            collectWrites(stmt.body, out);
        }
    }
    return out;
}
// ── Reduction-loop detection ──────────────────────────────────────────────────
/**
 * A loop is a "reduction loop" when every direct Assign in its body uses
 * `accumulate = true`.  The matmul k-loop satisfies this; i and j do not.
 * Used by the tiling pass to skip reduction dimensions by default.
 */
function isReductionLoop(loop) {
    const directAssigns = loop.body.filter(s => s.kind === "Assign");
    if (directAssigns.length === 0)
        return false;
    return directAssigns.every(a => a.accumulate);
}
// ── Expression and statement substitution ─────────────────────────────────────
/**
 * Recursively substitute loop variable names in an expression.
 * `subst` maps variable names to replacement LoopExprs.
 */
function substituteExpr(expr, subst) {
    switch (expr.kind) {
        case "LoopVar":
            return subst.get(expr.name) ?? expr;
        case "Literal":
            return expr;
        case "MemRef":
            return (0, loopIR_1.memRef)(expr.buffer, expr.indices.map(i => substituteExpr(i, subst)));
        case "BinOp":
            return (0, loopIR_1.binOp)(expr.op, substituteExpr(expr.lhs, subst), substituteExpr(expr.rhs, subst));
        case "CallBuiltin":
            return (0, loopIR_1.callBuiltin)(expr.callee, expr.args.map(a => substituteExpr(a, subst)));
    }
}
/**
 * Recursively substitute loop variable names in a statement.
 *
 * Scoping rule: a ForLoop's own induction variable shadows outer substitutions
 * inside its body.  The `hiExpr` (if any) is evaluated in the *outer* scope and
 * therefore uses the original `subst`.
 */
function substituteStmt(stmt, subst) {
    if (stmt.kind === "Assign") {
        return (0, loopIR_1.assign)(substituteExpr(stmt.target, subst), substituteExpr(stmt.value, subst), stmt.accumulate);
    }
    // hiExpr is in the outer scope — use original subst.
    const newHiExpr = stmt.hiExpr ? substituteExpr(stmt.hiExpr, subst) : undefined;
    // Remove the loop's own variable from the substitution for its body.
    let bodySubst = subst;
    if (subst.has(stmt.var.name)) {
        const m = new Map(subst);
        m.delete(stmt.var.name);
        bodySubst = m;
    }
    const newBody = stmt.body.map(s => substituteStmt(s, bodySubst));
    if (newHiExpr !== undefined) {
        return (0, loopIR_1.forLoopDyn)(stmt.var.name, stmt.lo, newHiExpr, newBody);
    }
    return (0, loopIR_1.forLoop)(stmt.var.name, stmt.lo, stmt.hi, newBody);
}
// ── Loop-nest reconstruction ──────────────────────────────────────────────────
/**
 * Reconstruct a ForLoop nest from outermost-to-innermost level descriptors,
 * wrapping the given innermost body.  Used by the fusion pass after merging
 * innermost bodies.
 */
function rebuildNest(levels, innerBody) {
    let body = [...innerBody];
    for (let d = levels.length - 1; d >= 0; d--) {
        const lv = levels[d];
        if (lv.hi === -1 && lv.hiExpr !== undefined) {
            body = [(0, loopIR_1.forLoopDyn)(lv.name, lv.lo, lv.hiExpr, body)];
        }
        else {
            body = [(0, loopIR_1.forLoop)(lv.name, lv.lo, lv.hi, body)];
        }
    }
    return body[0];
}
// ── Strip-mining (tiling primitive) ──────────────────────────────────────────
/**
 * Strip-mine one ForLoop dimension into outer and inner tile loops.
 *
 * Given:
 *   `for v in [lo, hi):`
 *
 * Produces:
 *   `for v_o in [0, ⌈(hi−lo)/T⌉):`
 *     `for v_i in [0, T):` (exact) or `[0, min(T, hi−lo − v_o·T)):` (edge tile)
 *       body with `v` replaced by `lo + v_o·T + v_i`
 *
 * The outer and inner variable names default to `${varName}_o` / `${varName}_i`.
 * Custom suffixes can be supplied via `outerSuffix` / `innerSuffix`.
 *
 * @throws if `loop.hi === -1` (dynamic bounds cannot be strip-mined statically).
 */
function stripMine(loop, tileSize, outerSuffix = "_o", innerSuffix = "_i") {
    if (loop.hi === -1) {
        throw new Error(`stripMine: cannot tile loop "${loop.var.name}" with dynamic bound (hi === -1).`);
    }
    const v = loop.var.name;
    const lo = loop.lo;
    const hi = loop.hi;
    const span = hi - lo;
    const outerName = `${v}${outerSuffix}`;
    const innerName = `${v}${innerSuffix}`;
    const outerV = (0, loopIR_1.loopVar)(outerName);
    const innerV = (0, loopIR_1.loopVar)(innerName);
    // Substitution: v → (lo === 0) ? outerV * T + innerV
    //                               : lo + outerV * T + innerV
    const tileOffset = (0, loopIR_1.binOp)("+", (0, loopIR_1.binOp)("*", outerV, (0, loopIR_1.literal)(tileSize)), innerV);
    const vExpr = lo === 0 ? tileOffset : (0, loopIR_1.binOp)("+", (0, loopIR_1.literal)(lo), tileOffset);
    const subst = new Map([[v, vExpr]]);
    const newBody = loop.body.map(s => substituteStmt(s, subst));
    const outerBound = Math.ceil(span / tileSize);
    const divisible = span % tileSize === 0;
    let innerLoop;
    if (divisible) {
        innerLoop = (0, loopIR_1.forLoop)(innerName, 0, tileSize, newBody);
    }
    else {
        // Edge-tile inner bound: min(T, span − v_o · T)
        const hiExpr = (0, loopIR_1.callBuiltin)("min", [
            (0, loopIR_1.literal)(tileSize),
            (0, loopIR_1.binOp)("-", (0, loopIR_1.literal)(span), (0, loopIR_1.binOp)("*", outerV, (0, loopIR_1.literal)(tileSize))),
        ]);
        innerLoop = (0, loopIR_1.forLoopDyn)(innerName, 0, hiExpr, newBody);
    }
    return (0, loopIR_1.forLoop)(outerName, 0, outerBound, [innerLoop]);
}
/**
 * Scan the top-level body of `fn` for adjacent ForLoop pairs that are safe
 * to fuse and return a structured analysis result.
 *
 * Safety conditions (all must hold):
 *   1. Both nests are perfect nests.
 *   2. Both nests have identical static iteration spaces.
 *   3. nest2 does not write any buffer that nest1 reads (no anti-dependence).
 *   4. nest1 and nest2 do not write the same buffer (no write-write conflict).
 */
function analyzeFusionCandidates(fn) {
    const candidates = [];
    const rejections = [];
    for (let i = 0; i + 1 < fn.body.length; i++) {
        const s1 = fn.body[i];
        const s2 = fn.body[i + 1];
        if (s1.kind !== "ForLoop" || s2.kind !== "ForLoop")
            continue;
        const n1 = extractPerfectNest(s1);
        const n2 = extractPerfectNest(s2);
        if (!n1) {
            rejections.push({ index1: i, index2: i + 1, reason: "First nest is not a perfect nest" });
            continue;
        }
        if (!n2) {
            rejections.push({ index1: i, index2: i + 1, reason: "Second nest is not a perfect nest" });
            continue;
        }
        if (!isSameIterSpace(n1, n2)) {
            const d1 = n1.levels.map(l => l.hi).join("×");
            const d2 = n2.levels.map(l => l.hi).join("×");
            rejections.push({
                index1: i, index2: i + 1,
                reason: `Incompatible iteration spaces: [${d1}] vs [${d2}]`,
            });
            continue;
        }
        const w1 = collectWrites([s1]);
        const w2 = collectWrites([s2]);
        const r1 = collectReads([s1]);
        const r2 = collectReads([s2]);
        // Anti-dependence check: nothing nest2 writes may be read by nest1.
        const antiDep = [...w2].filter(b => r1.has(b));
        if (antiDep.length > 0) {
            rejections.push({
                index1: i, index2: i + 1,
                reason: `Anti-dependence on buffer(s): [${antiDep.join(", ")}]`,
            });
            continue;
        }
        // Write-write conflict: nest1 and nest2 may not write the same buffer.
        const wwConflict = [...w1].filter(b => w2.has(b));
        if (wwConflict.length > 0) {
            rejections.push({
                index1: i, index2: i + 1,
                reason: `Write-write conflict on buffer(s): [${wwConflict.join(", ")}]`,
            });
            continue;
        }
        // Producer-consumer buffers: written by nest1, read by nest2.
        const sharedBuffers = [...w1].filter(b => r2.has(b));
        candidates.push({ index1: i, index2: i + 1, nest1: n1, nest2: n2, sharedBuffers });
    }
    return { candidates, rejections };
}
/**
 * Scan the top-level body of `fn` for ForLoops that are eligible for tiling.
 *
 * Eligibility criteria:
 *   1. Static bound (hi !== -1) on the outermost loop.
 *   2. Outermost loop span >= `minBound`.
 *   3. Not a reduction loop, unless `tileReductions` is true.
 *
 * Only reports top-level (outermost) loop eligibility; the tiling pass
 * recurses into inner loops during the rewrite.
 */
function analyzeTilingCandidates(fn, minBound, tileReductions) {
    const candidates = [];
    const rejections = [];
    for (let i = 0; i < fn.body.length; i++) {
        const stmt = fn.body[i];
        if (stmt.kind !== "ForLoop")
            continue;
        if (stmt.hi === -1) {
            rejections.push({ bodyIndex: i, reason: `Dynamic bound on "${stmt.var.name}" — cannot tile` });
            continue;
        }
        const span = stmt.hi - stmt.lo;
        if (span < minBound) {
            rejections.push({
                bodyIndex: i,
                reason: `Loop "${stmt.var.name}" span ${span} < minBound ${minBound}`,
            });
            continue;
        }
        if (!tileReductions && isReductionLoop(stmt)) {
            rejections.push({
                bodyIndex: i,
                reason: `"${stmt.var.name}" is a reduction loop (tileReductions=false)`,
            });
            continue;
        }
        candidates.push({ bodyIndex: i, loop: stmt });
    }
    return { candidates, rejections };
}
//# sourceMappingURL=loopAnalysis.js.map