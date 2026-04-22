import { Graph } from "../ir/graph";
import { Pass, PassResult } from "./pass";
import { LayoutRuleRegistry } from "../patterns/layoutRules";
import { OpContractRegistry } from "../ops/opContracts";
export declare class LayoutTransformPass implements Pass {
    private readonly ruleRegistry;
    private readonly opRegistry;
    readonly name = "LayoutTransformPass";
    constructor(ruleRegistry: LayoutRuleRegistry, opRegistry?: OpContractRegistry);
    run(graph: Graph): PassResult;
    private _applyLayoutRewrite;
    private _applyCancellation;
    private _applyPropagation;
}
