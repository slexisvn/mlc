// ─────────────────────────────────────────────────────────────────────────────
// passes/loopFusionPass.ts
//
// Loop-level fusion pass.
//
// Algorithm
// ─────────
// For each LoopFunction in the module:
//   1. Run analyzeFusionCandidates() to find adjacent perfect-nest pairs that
//      are safe to fuse (same iteration space, no dependence violations).
//   2. Apply each approved pair greedily, left-to-right, skipping any pair
//      whose second nest was already consumed by an earlier fusion.
//   3. For each fusion:
//        a. Build a variable substitution map (nest2 vars → nest1 vars).
//        b. Apply the substitution to nest2's innermost Assign body.
//        c. Merge: fused inner body = nest1.innerBody ++ substituted nest2.innerBody.
//        d. Reconstruct the outer ForLoop nest using nest1's level descriptors.
//   4. Replace fn.body with the compacted result.
//
// Safety
// ──────
// • Only perfect nests are fused — matmul/linear_relu nests (non-perfect)
//   are left untouched.
// • Anti-dependence and write-write conflict checks are enforced by the
//   analysis layer before this pass acts.
// • The resulting LoopModule is validated by the LoopPassManager if
//   validateAfterEachPass is enabled.
// ─────────────────────────────────────────────────────────────────────────────

import { LoopModule, LoopFunction, LoopStmt, Assign, loopVar } from "../ir/loopIR";
import { PassLog }                                               from "./pass";
import { LoopPass, LoopPassResult }                             from "./loopPass";
import {
  PerfectNestInfo,
  FusionCandidate,
  analyzeFusionCandidates,
  rebuildNest,
  substituteStmt,
} from "../analysis/loopAnalysis";

// ── Configuration ─────────────────────────────────────────────────────────────

export interface LoopFusionConfig {
  /**
   * Run the fusion scan repeatedly until no more pairs can be fused.
   * Useful when three or more adjacent nests benefit from cascaded fusion.
   * Default: true.
   */
  readonly iterateToFixpoint: boolean;
}

export const DEFAULT_LOOP_FUSION_CONFIG: LoopFusionConfig = {
  iterateToFixpoint: true,
};

// ── Pass ─────────────────────────────────────────────────────────────────────

export class LoopFusionPass implements LoopPass {
  readonly name = "LoopFusionPass";

  private readonly config: LoopFusionConfig;

  constructor(config: Partial<LoopFusionConfig> = {}) {
    this.config = { ...DEFAULT_LOOP_FUSION_CONFIG, ...config };
  }

  run(module: LoopModule): LoopPassResult {
    const logs:    PassLog[]      = [];
    let   changed                 = false;
    const newFunctions: LoopFunction[] = [];

    for (const fn of module.functions) {
      const { fn: newFn, changed: fnChanged, logs: fnLogs } =
        this._fuseFunction(fn);
      newFunctions.push(newFn);
      logs.push(...fnLogs);
      if (fnChanged) changed = true;
    }

    if (!changed) {
      logs.push({ level: "info", message: "No loop fusion opportunities found." });
      return { module, changed: false, logs };
    }

    const newModule: LoopModule = {
      graphId:     module.graphId,
      functions:   newFunctions,
      diagnostics: module.diagnostics,
    };

    return { module: newModule, changed: true, logs };
  }

  // ── Per-function fusion ────────────────────────────────────────────────────

  private _fuseFunction(
    fn: LoopFunction,
  ): { fn: LoopFunction; changed: boolean; logs: PassLog[] } {
    const logs: PassLog[] = [];
    let   body             = [...fn.body];
    let   totalFused       = 0;

    // Iterate to fixed point if configured; otherwise a single pass.
    for (;;) {
      const analysis   = analyzeFusionCandidates({ ...fn, body });
      const { newBody, fusedCount } = _applyFusions(body, analysis.candidates, fn.name, logs);

      for (const rej of analysis.rejections) {
        logs.push({
          level:   "info",
          message: `  fn=${fn.name} pair(${rej.index1},${rej.index2}) rejected: ${rej.reason}`,
        });
      }

      body        = newBody;
      totalFused += fusedCount;

      if (!this.config.iterateToFixpoint || fusedCount === 0) break;
    }

    if (totalFused === 0) {
      return { fn, changed: false, logs };
    }

    logs.push({
      level:   "info",
      message: `fn=${fn.name}: fused ${totalFused} nest pair(s).`,
    });

    return {
      fn:      { ...fn, body },
      changed: true,
      logs,
    };
  }
}

// ── Pure rewrite helpers ──────────────────────────────────────────────────────

/**
 * Apply approved fusion candidates to a body array.
 * Returns the new body and the count of pairs actually fused.
 */
function _applyFusions(
  body:       LoopStmt[],
  candidates: readonly FusionCandidate[],
  fnName:     string,
  logs:       PassLog[],
): { newBody: LoopStmt[]; fusedCount: number } {
  if (candidates.length === 0) return { newBody: body, fusedCount: 0 };

  // Track which body indices have been consumed (second member of a fused pair).
  const consumed    = new Set<number>();
  const replacements = new Map<number, LoopStmt>();
  let   fusedCount   = 0;

  for (const cand of candidates) {
    if (consumed.has(cand.index1) || consumed.has(cand.index2)) continue;

    const fused = _mergeNests(cand.nest1, cand.nest2);
    const ops1  = cand.nest1.innerBody.map(a => a.target.buffer).join(", ");
    const ops2  = cand.nest2.innerBody.map(a => a.target.buffer).join(", ");
    const dims  = cand.nest1.levels.map(l => l.hi).join("×");

    logs.push({
      level:   "info",
      message:
        `  fn=${fnName}: fusing nests@(${cand.index1},${cand.index2}) ` +
        `[${dims}] — writes=[${ops1}] ⊕ writes=[${ops2}]` +
        (cand.sharedBuffers.length > 0
          ? ` via shared=[${cand.sharedBuffers.join(", ")}]`
          : ""),
    });

    replacements.set(cand.index1, fused);
    consumed.add(cand.index2);
    fusedCount++;
  }

  const newBody: LoopStmt[] = [];
  for (let i = 0; i < body.length; i++) {
    if (consumed.has(i)) continue;
    newBody.push(replacements.get(i) ?? body[i]);
  }

  return { newBody, fusedCount };
}

/**
 * Merge two compatible perfect nests into one.
 *
 * Variable renaming: nest2's induction variables are substituted with nest1's
 * so the fused nest uses a single consistent set of variable names.
 */
function _mergeNests(n1: PerfectNestInfo, n2: PerfectNestInfo): LoopStmt {
  // Build substitution: n2 var → n1 var (only where names differ).
  const subst = new Map<string, import("../ir/loopIR").LoopExpr>();
  for (let d = 0; d < n1.levels.length; d++) {
    if (n1.levels[d].name !== n2.levels[d].name) {
      subst.set(n2.levels[d].name, loopVar(n1.levels[d].name));
    }
  }

  const n2Body: readonly Assign[] =
    subst.size > 0
      ? (n2.innerBody.map(a => substituteStmt(a, subst)) as Assign[])
      : n2.innerBody;

  const fusedInnerBody: LoopStmt[] = [...n1.innerBody, ...n2Body];
  return rebuildNest(n1.levels, fusedInnerBody);
}
