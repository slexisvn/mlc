import { GraphIR }                              from "../ir/schema";
import { TensorId }                             from "../ir/ids";
import { OpSchemaRegistry, defaultOpRegistry }  from "../core/opRegistry";
import { analyzeAutodiff }                      from "./backwardAnalysis";
import { BackwardGraphBuilder }                 from "./backwardBuilder";
import { GradRegistry, defaultGradRegistry }    from "./gradRegistry";

export type { BackwardResult, GradBuilderFn, GradContext } from "./types";
export { GradRegistry, defaultGradRegistry }               from "./gradRegistry";
export { DEFAULT_GRAD_BUILDERS }                           from "./gradBuilders";

import type { BackwardResult } from "./types";

export function buildBackwardGraph(
  fwd:          GraphIR,
  rootIds:      readonly TensorId[],
  paramIds:     readonly TensorId[],
  opRegistry:   OpSchemaRegistry = defaultOpRegistry,
  gradRegistry: GradRegistry     = defaultGradRegistry,
): BackwardResult {
  const analysis = analyzeAutodiff(fwd, paramIds, rootIds);
  const builder  = new BackwardGraphBuilder(fwd, analysis, opRegistry, gradRegistry);
  return builder.build(rootIds, paramIds);
}


