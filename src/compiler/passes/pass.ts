// ─────────────────────────────────────────────────────────────────────────────
// passes/pass.ts
//
// Core Pass abstraction used by every optimization pass in the pipeline.
//
// Design decisions
// ────────────────
// • run() accepts and returns a Graph.  The contract is:
//     - Return the SAME graph object if no changes were made (changed: false).
//     - Return a new / mutated clone if the graph was modified (changed: true).
//   This lets the PassManager track dirty state cheaply.
//
// • PassResult carries structured logs rather than side-effecting console calls
//   so the PassManager can route them however the caller wants (console, file,
//   structured JSON, etc.).
//
// • Adding a new pass = implement Pass + call passManager.addPass(new MyPass()).
//   No other change required.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph } from "../ir/graph";

/** Severity levels for pass diagnostic messages. */
export type LogLevel = "info" | "warn" | "error";

/** A single diagnostic entry emitted by a pass. */
export interface PassLog {
  level:   LogLevel;
  message: string;
}

/** Returned by every pass after execution. */
export interface PassResult {
  /** The (possibly transformed) graph. */
  graph:   Graph;
  /** Whether the graph was structurally modified. */
  changed: boolean;
  /** All diagnostic messages emitted during this pass run. */
  logs:    PassLog[];
}

/**
 * Base interface for all compiler passes.
 *
 * Example minimal pass:
 *
 *   class NoOpPass implements Pass {
 *     readonly name = "NoOpPass";
 *     run(graph: Graph): PassResult {
 *       return { graph, changed: false, logs: [] };
 *     }
 *   }
 */
export interface Pass {
  readonly name: string;
  run(graph: Graph): PassResult;
}
