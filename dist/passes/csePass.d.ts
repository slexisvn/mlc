import { Graph } from "../ir/graph";
import { Pass, PassResult } from "./pass";
import { OpContractRegistry } from "../ops/opContracts";
export declare class CSEPass implements Pass {
    private readonly opRegistry;
    readonly name = "CSEPass";
    constructor(opRegistry?: OpContractRegistry);
    run(graph: Graph): PassResult;
}
