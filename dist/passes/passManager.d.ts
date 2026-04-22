import { Graph } from "../ir/graph";
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
    logSink?: (entry: PassLog & {
        passName: string;
    }) => void;
}
export declare class PassManager {
    private readonly passes;
    private readonly options;
    constructor(options?: Partial<PassManagerOptions>);
    /** Register a pass.  Returns `this` for method chaining. */
    addPass(pass: Pass): this;
    /**
     * Register multiple passes at once.  Passes are appended in the supplied
     * order.  Returns `this` for method chaining.
     *
     * Useful when configuring a pipeline from a pre-built list:
     *   pm.addPasses(layoutPass, fusionPass, loopLoweringPass);
     */
    addPasses(...passes: Pass[]): this;
    /**
     * Run all registered passes sequentially on `inputGraph`.
     *
     * @returns The final (optimised) graph after all passes.
     * @throws  If validateAfterEachPass=true and any pass leaves the graph invalid.
     */
    run(inputGraph: Graph): Graph;
    private _emit;
}
