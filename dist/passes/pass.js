"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
//# sourceMappingURL=pass.js.map