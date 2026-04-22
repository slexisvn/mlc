import { Graph } from "../ir/graph";
import { Pass, PassResult } from "./pass";
import { RuleRegistry } from "../patterns/rules";
import { CostModel } from "../optimizer/costModel";
export declare class FusionPass implements Pass {
    private readonly registry;
    private readonly costModel;
    readonly name = "FusionPass";
    constructor(registry: RuleRegistry, costModel: CostModel);
    run(graph: Graph): PassResult;
    private _applyFusion;
}
