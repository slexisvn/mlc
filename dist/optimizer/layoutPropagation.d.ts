import { Graph } from "../ir/graph";
import { LayoutFormat } from "../ir/layouts";
import { OpContractRegistry } from "../ops/opContracts";
export interface PropagationResult {
    readonly canPropagate: boolean;
    /** Id of the first node that blocked propagation, when canPropagate=false. */
    readonly blockedAt?: string;
    readonly blockedOp?: string;
    readonly reason?: string;
}
/**
 * Test whether `layout` can flow through every node in `chain` without
 * requiring an explicit conversion.
 *
 * @param chain     Ordered node ids to test (need not include boundary nodes).
 * @param layout    The layout being propagated.
 * @param graph     Graph containing the nodes.
 * @param registry  Op contract registry; defaults to DEFAULT_CONTRACT_REGISTRY.
 */
export declare function canPropagateLayout(chain: string[], layout: LayoutFormat, graph: Graph, registry?: OpContractRegistry): PropagationResult;
/**
 * True when the given op accepts the layout without needing a conversion.
 * Agnostic / preserving / unregistered ops always return true.
 * Transforming ops always return false (they change the layout).
 * Sensitive ops return true iff the layout is in their required-inputs list
 * or the list is empty.
 */
export declare function opAcceptsLayout(op: string, layout: LayoutFormat, registry?: OpContractRegistry): boolean;
