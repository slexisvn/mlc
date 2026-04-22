import { LayoutFormat } from "../ir/layouts";
export type OpLayoutBehavior = "agnostic" | "preserving" | "sensitive" | "transforming";
export type FusibilityClass = "fusible" | "unfusible" | "conditional";
export interface OpContract {
    readonly op: string;
    readonly fusibilityClass: FusibilityClass;
    readonly layoutBehavior: OpLayoutBehavior;
    readonly requiredInputLayouts?: readonly LayoutFormat[];
    readonly outputLayout?: LayoutFormat;
    readonly pure?: boolean;
    readonly foldable?: boolean;
    readonly description?: string;
}
export declare class OpContractRegistry {
    private readonly _contracts;
    constructor(initial?: readonly OpContract[]);
    register(contract: OpContract): this;
    get(op: string): OpContract | undefined;
    has(op: string): boolean;
    isLayoutAgnostic(op: string): boolean;
    isLayoutSensitive(op: string): boolean;
    isLayoutTransforming(op: string): boolean;
    isLayoutPreserving(op: string): boolean;
    isFusible(op: string): boolean;
    isPure(op: string): boolean;
    isFoldable(op: string): boolean;
    getAll(): readonly OpContract[];
}
export declare const DEFAULT_OP_CONTRACTS: readonly OpContract[];
export declare const DEFAULT_CONTRACT_REGISTRY: OpContractRegistry;
