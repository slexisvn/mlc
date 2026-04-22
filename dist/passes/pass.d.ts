import { Graph } from "../ir/graph";
/** Severity levels for pass diagnostic messages. */
export type LogLevel = "info" | "warn" | "error";
/** A single diagnostic entry emitted by a pass. */
export interface PassLog {
    level: LogLevel;
    message: string;
}
/** Returned by every pass after execution. */
export interface PassResult {
    /** The (possibly transformed) graph. */
    graph: Graph;
    /** Whether the graph was structurally modified. */
    changed: boolean;
    /** All diagnostic messages emitted during this pass run. */
    logs: PassLog[];
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
