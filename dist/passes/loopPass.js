"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// passes/loopPass.ts
//
// LoopPass interface and LoopPassManager — the Loop IR counterpart of the
// graph-level Pass / PassManager pair.
//
// Design
// ──────
// • LoopPass.run() accepts a LoopModule and returns a new LoopModule plus a
//   structured log.  The contract mirrors the graph Pass contract:
//     - Return the SAME module object if nothing changed (changed: false).
//     - Return a fresh module object if the module was transformed (changed: true).
// • LoopPassManager runs passes sequentially and optionally validates the
//   LoopModule after each pass using validateLoopModule().
// • Logs are routed through an optional logSink (same signature as PassManager).
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.LoopPassManager = void 0;
const validateLoop_1 = require("../ir/validateLoop");
const DEFAULT_OPTIONS = {
    validateAfterEachPass: true,
};
// ── LoopPassManager ───────────────────────────────────────────────────────────
class LoopPassManager {
    constructor(options = {}) {
        this.passes = [];
        this.options = { ...DEFAULT_OPTIONS, ...options };
    }
    /** Register a pass.  Returns `this` for method chaining. */
    addPass(pass) {
        this.passes.push(pass);
        return this;
    }
    /** Register multiple passes at once.  Returns `this` for method chaining. */
    addPasses(...passes) {
        for (const p of passes)
            this.passes.push(p);
        return this;
    }
    /**
     * Run all registered passes sequentially on `inputModule`.
     *
     * @returns  The final (optimised) LoopModule.
     * @throws   If validateAfterEachPass=true and any pass leaves the module invalid.
     */
    run(inputModule) {
        let module = inputModule;
        for (const pass of this.passes) {
            this._emit(pass.name, "info", `${"─".repeat(50)}`);
            this._emit(pass.name, "info", `Running loop pass: ${pass.name}`);
            const result = pass.run(module);
            for (const log of result.logs) {
                this._emit(pass.name, log.level, log.message);
            }
            this._emit(pass.name, "info", result.changed
                ? `LoopModule modified by "${pass.name}"`
                : `No changes in "${pass.name}"`);
            module = result.module;
            if (this.options.validateAfterEachPass) {
                const vr = (0, validateLoop_1.validateLoopModule)(module);
                // Surface warnings even when valid.
                for (const w of vr.warnings) {
                    this._emit(pass.name, "warn", `[${w.kind}] fn=${w.fn}: ${w.message}`);
                }
                if (vr.valid) {
                    this._emit(pass.name, "info", `LoopModule valid after "${pass.name}" ✓`);
                }
                else {
                    for (const e of vr.errors) {
                        this._emit(pass.name, "error", `[${e.kind}] fn=${e.fn}: ${e.message}`);
                    }
                    throw new Error(`LoopModule validation failed after pass "${pass.name}". ` +
                        `See logs for details.`);
                }
            }
        }
        return module;
    }
    // ── Private ────────────────────────────────────────────────────────────────
    _emit(passName, level, message) {
        const entry = { passName, level, message };
        if (this.options.logSink) {
            this.options.logSink(entry);
        }
        else {
            console.log(`[${passName}][${level.toUpperCase()}] ${message}`);
        }
    }
}
exports.LoopPassManager = LoopPassManager;
//# sourceMappingURL=loopPass.js.map