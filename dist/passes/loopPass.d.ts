import { LoopModule } from "../ir/loopIR";
import { PassLog } from "./pass";
export interface LoopPassResult {
    /** The (possibly transformed) module. */
    module: LoopModule;
    /** Whether the module was structurally modified. */
    changed: boolean;
    /** Diagnostic messages emitted during this pass run. */
    logs: PassLog[];
}
/** Base interface for all Loop IR optimisation passes. */
export interface LoopPass {
    readonly name: string;
    run(module: LoopModule): LoopPassResult;
}
export interface LoopPassManagerOptions {
    /**
     * Validate LoopModule invariants after every pass.
     * Recommended during development.  Default: true.
     */
    validateAfterEachPass: boolean;
    /**
     * Optional log sink; receives all log entries annotated with pass name.
     * When absent, entries are printed to console.
     */
    logSink?: (entry: PassLog & {
        passName: string;
    }) => void;
}
export declare class LoopPassManager {
    private readonly passes;
    private readonly options;
    constructor(options?: Partial<LoopPassManagerOptions>);
    /** Register a pass.  Returns `this` for method chaining. */
    addPass(pass: LoopPass): this;
    /** Register multiple passes at once.  Returns `this` for method chaining. */
    addPasses(...passes: LoopPass[]): this;
    /**
     * Run all registered passes sequentially on `inputModule`.
     *
     * @returns  The final (optimised) LoopModule.
     * @throws   If validateAfterEachPass=true and any pass leaves the module invalid.
     */
    run(inputModule: LoopModule): LoopModule;
    private _emit;
}
