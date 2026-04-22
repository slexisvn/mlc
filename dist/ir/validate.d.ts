import { Graph } from "./graph";
export type ValidationErrorKind = "DanglingEdge" | "SSAViolation" | "MissingOutput" | "Cycle" | "OrphanNode" | "LayoutContractViolation";
export interface ValidationError {
    kind: ValidationErrorKind;
    message: string;
}
export interface ValidationResult {
    valid: boolean;
    errors: ValidationError[];
}
export declare function validateGraph(graph: Graph): ValidationResult;
