// ─────────────────────────────────────────────────────────────────────────────
// passes/passManager.ts
//
// Sequential pass pipeline with optional inter-pass graph validation.
//
// Usage:
//   const pm = new PassManager({ validateAfterEachPass: true });
//   pm.addPass(new FusionPass(registry, costModel));
//   const optimised = pm.run(inputGraph);
//
// The PassManager does NOT own the passes; it borrows them.
// Passes can be shared across multiple managers.
// ─────────────────────────────────────────────────────────────────────────────

import { Graph } from "../ir/graph";
import { validateGraph } from "../ir/validate";
import { Pass, PassLog } from "./pass";

export interface PassManagerOptions {
  /**
   * Validate graph invariants after every pass.
   * Recommended during development; can be disabled for production speed.
   * Default: true.
   */
  validateAfterEachPass: boolean;

  /**
   * Optional log sink.  If provided, all log entries are forwarded here instead
   * of being printed to console.  Useful for capturing logs in tests.
   */
  logSink?: (entry: PassLog & { passName: string }) => void;
}

const DEFAULT_OPTIONS: PassManagerOptions = {
  validateAfterEachPass: true,
};

export class PassManager {
  private readonly passes:  Pass[]              = [];
  private readonly options: PassManagerOptions;

  constructor(options: Partial<PassManagerOptions> = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /** Register a pass.  Returns `this` for method chaining. */
  addPass(pass: Pass): this {
    this.passes.push(pass);
    return this;
  }

  /**
   * Register multiple passes at once.  Passes are appended in the supplied
   * order.  Returns `this` for method chaining.
   *
   * Useful when configuring a pipeline from a pre-built list:
   *   pm.addPasses(layoutPass, fusionPass, loopLoweringPass);
   */
  addPasses(...passes: Pass[]): this {
    for (const p of passes) this.passes.push(p);
    return this;
  }

  /**
   * Run all registered passes sequentially on `inputGraph`.
   *
   * @returns The final (optimised) graph after all passes.
   * @throws  If validateAfterEachPass=true and any pass leaves the graph invalid.
   */
  run(inputGraph: Graph): Graph {
    let graph = inputGraph;

    for (const pass of this.passes) {
      this._emit(pass.name, "info", `${"─".repeat(50)}`);
      this._emit(pass.name, "info", `Running pass: ${pass.name}`);

      const result = pass.run(graph);

      // Forward all pass-internal logs.
      for (const log of result.logs) {
        this._emit(pass.name, log.level, log.message);
      }

      this._emit(
        pass.name, "info",
        result.changed ? `Graph modified by "${pass.name}"` : `No changes in "${pass.name}"`,
      );

      graph = result.graph;

      // Optional post-pass validation gate.
      if (this.options.validateAfterEachPass) {
        const vr = validateGraph(graph);
        if (vr.valid) {
          this._emit(pass.name, "info", `Graph valid after "${pass.name}" ✓`);
        } else {
          for (const err of vr.errors) {
            this._emit(pass.name, "error", `[${err.kind}] ${err.message}`);
          }
          throw new Error(
            `Graph validation failed after pass "${pass.name}". ` +
            `See logs for details.`,
          );
        }
      }
    }

    return graph;
  }

  // ─── Private ──────────────────────────────────────────────────────────────

  private _emit(passName: string, level: PassLog["level"], message: string): void {
    const entry = { passName, level, message };
    if (this.options.logSink) {
      this.options.logSink(entry);
    } else {
      const tag = `[${passName}][${level.toUpperCase()}]`;
      console.log(`${tag} ${message}`);
    }
  }
}
