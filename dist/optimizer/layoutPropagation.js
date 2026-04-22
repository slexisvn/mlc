"use strict";
// ─────────────────────────────────────────────────────────────────────────────
// optimizer/layoutPropagation.ts
//
// Utilities for testing whether a layout can propagate through a sequence of
// ops without hitting a layout boundary.
//
// "Propagation" means: every op in the chain either does not care about layout
// (agnostic) or preserves the layout of its primary input (preserving).
// A layout-sensitive op with incompatible requirements, or a layout-transforming
// op, blocks propagation.
//
// These helpers are used by LayoutTransformPass to validate that a "sandwich"
// rewrite (removing outer transposes from transpose→op→transpose) is safe:
// the middle op must be compatible with the original layout.
// ─────────────────────────────────────────────────────────────────────────────
Object.defineProperty(exports, "__esModule", { value: true });
exports.canPropagateLayout = canPropagateLayout;
exports.opAcceptsLayout = opAcceptsLayout;
const layouts_1 = require("../ir/layouts");
const opContracts_1 = require("../ops/opContracts");
// ─── Public API ───────────────────────────────────────────────────────────────
/**
 * Test whether `layout` can flow through every node in `chain` without
 * requiring an explicit conversion.
 *
 * @param chain     Ordered node ids to test (need not include boundary nodes).
 * @param layout    The layout being propagated.
 * @param graph     Graph containing the nodes.
 * @param registry  Op contract registry; defaults to DEFAULT_CONTRACT_REGISTRY.
 */
function canPropagateLayout(chain, layout, graph, registry = opContracts_1.DEFAULT_CONTRACT_REGISTRY) {
    for (const nid of chain) {
        const node = graph.getNode(nid);
        const contract = registry.get(node.op);
        // Agnostic or unregistered: always compatible.
        if (contract === undefined ||
            contract.layoutBehavior === "agnostic" ||
            contract.layoutBehavior === "preserving") {
            continue;
        }
        if (contract.layoutBehavior === "transforming") {
            return {
                canPropagate: false,
                blockedAt: nid,
                blockedOp: node.op,
                reason: `Op "${node.op}" is a layout-transforming op; propagation stops here.`,
            };
        }
        if (contract.layoutBehavior === "sensitive") {
            const required = contract.requiredInputLayouts ?? [];
            if (required.length > 0 &&
                !required.includes(layout) &&
                !required.includes(layouts_1.Layouts.ANY)) {
                return {
                    canPropagate: false,
                    blockedAt: nid,
                    blockedOp: node.op,
                    reason: `Op "${node.op}" requires layout [${required.join("|")}] ` +
                        `but propagating layout is "${layout}".`,
                };
            }
        }
    }
    return { canPropagate: true };
}
/**
 * True when the given op accepts the layout without needing a conversion.
 * Agnostic / preserving / unregistered ops always return true.
 * Transforming ops always return false (they change the layout).
 * Sensitive ops return true iff the layout is in their required-inputs list
 * or the list is empty.
 */
function opAcceptsLayout(op, layout, registry = opContracts_1.DEFAULT_CONTRACT_REGISTRY) {
    const contract = registry.get(op);
    if (!contract)
        return true;
    if (contract.layoutBehavior === "agnostic" ||
        contract.layoutBehavior === "preserving")
        return true;
    if (contract.layoutBehavior === "transforming")
        return false;
    // sensitive
    const required = contract.requiredInputLayouts ?? [];
    return required.length === 0 ||
        required.includes(layout) ||
        required.includes(layouts_1.Layouts.ANY);
}
//# sourceMappingURL=layoutPropagation.js.map