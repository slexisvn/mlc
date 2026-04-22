import { Graph } from "../ir/graph";
import { Pass, PassResult } from "./pass";
import { OpContractRegistry } from "../ops/opContracts";
/** Reset the const-node id counter (for deterministic test output). */
export declare function resetCFCounter(): void;
export declare class ConstantFoldingPass implements Pass {
    private readonly opRegistry;
    readonly name = "ConstantFoldingPass";
    constructor(opRegistry?: OpContractRegistry);
    run(graph: Graph): PassResult;
}
