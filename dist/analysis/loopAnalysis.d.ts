import { ForLoop, LoopStmt, LoopExpr, Assign, LoopFunction } from "../ir/loopIR";
/** One nesting level descriptor extracted from a perfect loop nest. */
export interface PerfectNestLevel {
    readonly name: string;
    readonly lo: number;
    /** Static upper bound, or -1 when dynamic (from strip-mined inner loops). */
    readonly hi: number;
    /** Present when hi === -1 (edge-tile bound expression). */
    readonly hiExpr?: LoopExpr;
}
/**
 * Description of a perfect nest — each intermediate level has exactly one
 * body statement (a ForLoop), and the innermost level contains only Assigns.
 */
export interface PerfectNestInfo {
    /** Ordered outermost → innermost level descriptors. */
    readonly levels: readonly PerfectNestLevel[];
    /** Assign statements at the innermost loop body. */
    readonly innerBody: readonly Assign[];
}
/**
 * If `loop` is a perfect nest, return its descriptor.  Otherwise return null.
 *
 * A perfect nest is defined as: every intermediate nesting level has exactly
 * one body statement (a ForLoop), and the innermost level has one or more
 * Assign statements and no ForLoops.
 */
export declare function extractPerfectNest(loop: ForLoop): PerfectNestInfo | null;
/** True iff `stmt` is a ForLoop forming a perfect nest. */
export declare function isPerfectNest(stmt: LoopStmt): stmt is ForLoop;
/**
 * Two perfect nests are compatible for fusion when they have the same number
 * of dimensions, the same static bounds at every level, and no dynamic bounds
 * (hi !== -1 at every level).
 */
export declare function isSameIterSpace(a: PerfectNestInfo, b: PerfectNestInfo): boolean;
/** Collect all buffer names referenced in an expression tree (read positions). */
export declare function collectExprBuffers(expr: LoopExpr, out?: Set<string>): Set<string>;
/**
 * Collect all buffer names that appear in read position within `stmts`.
 * Includes MemRefs in Assign value expressions and in index expressions of
 * Assign targets, plus hiExpr references in ForLoop headers.
 */
export declare function collectReads(stmts: readonly LoopStmt[], out?: Set<string>): Set<string>;
/**
 * Collect all buffer names that appear as Assign target buffers in `stmts`
 * (i.e., buffers that are written).
 */
export declare function collectWrites(stmts: readonly LoopStmt[], out?: Set<string>): Set<string>;
/**
 * A loop is a "reduction loop" when every direct Assign in its body uses
 * `accumulate = true`.  The matmul k-loop satisfies this; i and j do not.
 * Used by the tiling pass to skip reduction dimensions by default.
 */
export declare function isReductionLoop(loop: ForLoop): boolean;
/**
 * Recursively substitute loop variable names in an expression.
 * `subst` maps variable names to replacement LoopExprs.
 */
export declare function substituteExpr(expr: LoopExpr, subst: ReadonlyMap<string, LoopExpr>): LoopExpr;
/**
 * Recursively substitute loop variable names in a statement.
 *
 * Scoping rule: a ForLoop's own induction variable shadows outer substitutions
 * inside its body.  The `hiExpr` (if any) is evaluated in the *outer* scope and
 * therefore uses the original `subst`.
 */
export declare function substituteStmt(stmt: LoopStmt, subst: ReadonlyMap<string, LoopExpr>): LoopStmt;
/**
 * Reconstruct a ForLoop nest from outermost-to-innermost level descriptors,
 * wrapping the given innermost body.  Used by the fusion pass after merging
 * innermost bodies.
 */
export declare function rebuildNest(levels: readonly PerfectNestLevel[], innerBody: readonly LoopStmt[]): ForLoop;
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
export declare function stripMine(loop: ForLoop, tileSize: number, outerSuffix?: string, innerSuffix?: string): ForLoop;
/** A pair of adjacent perfect nests that are safe to fuse. */
export interface FusionCandidate {
    /** Index of the first (earlier) nest in `fn.body`. */
    readonly index1: number;
    /** Index of the second (later) nest in `fn.body`. */
    readonly index2: number;
    readonly nest1: PerfectNestInfo;
    readonly nest2: PerfectNestInfo;
    /** Buffer names written by nest1 and read by nest2 (producer-consumer flow). */
    readonly sharedBuffers: readonly string[];
}
export interface FusionRejection {
    readonly index1: number;
    readonly index2: number;
    readonly reason: string;
}
export interface LoopFusionAnalysis {
    readonly candidates: readonly FusionCandidate[];
    readonly rejections: readonly FusionRejection[];
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
export declare function analyzeFusionCandidates(fn: LoopFunction): LoopFusionAnalysis;
export interface TilingCandidate {
    /** 0-based index into `fn.body`. */
    readonly bodyIndex: number;
    readonly loop: ForLoop;
}
export interface TilingRejection {
    readonly bodyIndex: number;
    readonly reason: string;
}
export interface LoopTilingAnalysis {
    readonly candidates: readonly TilingCandidate[];
    readonly rejections: readonly TilingRejection[];
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
export declare function analyzeTilingCandidates(fn: LoopFunction, minBound: number, tileReductions: boolean): LoopTilingAnalysis;
