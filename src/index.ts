// ─────────────────────────────────────────────────────────────────────────────
// index.ts — public API surface + demo runner
//
// src/ contains exactly two subtrees:
//   framework/  — PyTorch-like model-definition API
//   compiler/   — ML compiler (passes, IR, analysis, optimiser)
// ─────────────────────────────────────────────────────────────────────────────

// ── Public namespaces ─────────────────────────────────────────────────────────
export * as framework from "./framework";
export * as compiler  from "./compiler";

// ─────────────────────────────────────────────────────────────────────────────
// Demo runner — executed only when this file is the program entry point.
// ─────────────────────────────────────────────────────────────────────────────
if (require.main === module) {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { runPytorchCompilerDemo } = require("./framework/examples/pytorchCompilerDemo");
  runPytorchCompilerDemo();
}