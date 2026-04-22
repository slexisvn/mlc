import { Graph } from "../ir/graph";
import { Pass, PassResult } from "./pass";
export declare class DeadCodeEliminationPass implements Pass {
    readonly name = "DeadCodeEliminationPass";
    run(graph: Graph): PassResult;
}
