import { LoopModule } from "./loopIR";
export type LoopValidationErrorKind = "InvalidBound" | "UndeclaredBuffer" | "FreeVariable" | "WriteToInput";
export interface LoopValidationError {
    readonly kind: LoopValidationErrorKind;
    readonly fn: string;
    readonly message: string;
}
export interface LoopValidationResult {
    readonly valid: boolean;
    /** Hard errors that indicate a malformed LoopModule. */
    readonly errors: readonly LoopValidationError[];
    /** Soft warnings (non-fatal anomalies). */
    readonly warnings: readonly LoopValidationError[];
}
export declare function validateLoopModule(module: LoopModule): LoopValidationResult;
