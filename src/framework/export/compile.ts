import { ExportSession } from "./session";
import { buildBackwardGraph } from "../autodiff";
import { importGraphIR } from "./importGraphIR";
import { validateIRPackage } from "../ir/validator";
import { validateGraph } from "../../compiler/ir/validate";
import { Module } from "../nn/module";
import { IRPackage } from "../ir/schema";
import { TensorId } from "../ir/ids";
import { SymbolicTensor } from "../tensor/tensor";
import { createDefaultPipeline } from "../../compiler/passes/pipelines";
import { printGraph, printDiff } from "../../compiler/debug/printer";
import { printLoopModule } from "../../compiler/debug/loopPrinter";

export interface CompileOptions {
  autodiff?: boolean;
  optimize?: boolean;
  logPasses?: boolean;
}

export function compile(options: CompileOptions = {}) {
  const autodiff = options.autodiff ?? false;
  const optimize = options.optimize ?? true;
  const logPasses = options.logPasses ?? true;

  return function <T extends { new(...args: any[]): Module }>(constructor: T): T {
    return class extends constructor {
      private _compiledPkg: IRPackage | null = null;

      forward(...args: any[]): any {
        // PyTorch Dynamo intercepts and caches
        if (this._compiledPkg) {
          if (logPasses) {
            console.log(`[Cache Hit] Returning previously compiled computation graph for ${constructor.name}`);
          }
          return this._compiledPkg;
        }

        const session = new ExportSession({ id: constructor.name.toLowerCase() });
        let lossId: TensorId | undefined;

        session.build((ctx) => {
          const symbolicArgs = args.map((arg, i) => {
            if (arg && arg.isEager) {
              return ctx.input(`inp_${i}`, arg.dtype, arg.shape);
            }
            throw new Error(`@compile requires EagerTensors as inputs. Argument ${i} is not eager.`);
          });

          const result = constructor.prototype.forward.apply(this, symbolicArgs);

          if (Array.isArray(result)) {
            const symResult = result as SymbolicTensor[];
            ctx.markOutput(...symResult);
            lossId = symResult[symResult.length - 1].id;
          } else {
            const symResult = result as SymbolicTensor;
            ctx.markOutput(symResult);
            lossId = symResult.id;
          }
        });

        const fwdPkg = session.export("forward");
        const fwdVal = validateIRPackage(fwdPkg);
        if (!fwdVal.valid) {
          throw new Error(`Forward validation failed: ${fwdVal.errors.map(e => e.message).join(", ")}`);
        }

        let fullPkg = fwdPkg;
        if (autodiff && lossId) {
          const graphIR = fwdPkg.graphs[0];

          // Exclude tensors that are constants from gradient tracking
          const constTensorIds = new Set(
            fwdPkg.parameters?.filter(p => p.isConst).map(p => p.tensorId) || []
          );

          const paramIds = graphIR.inputIds.filter(tid =>
            graphIR.tensors[tid].isParam && !constTensorIds.has(tid)
          );

          const { backwardGraph, gradMap } = buildBackwardGraph(graphIR, [lossId], paramIds);
          fullPkg = { ...fwdPkg, graphs: [graphIR, backwardGraph] };

          if (logPasses) {
            console.log(`  ✓ Backward graph built — ${Object.keys(backwardGraph.nodes).length} nodes, ${gradMap.size} gradient(s)`);
          }
        }

        if (optimize) {
          runPipeline(fullPkg, logPasses);
        }

        this._compiledPkg = fullPkg;
        return fullPkg;
      }
    };
  };
}

function runPipeline(pkg: IRPackage, logPasses: boolean) {
  const SEP = "═".repeat(66);
  const sec = (title: string) => logPasses && console.log(`\n${SEP}\n  ${title}\n${SEP}`);

  for (const graphIR of pkg.graphs) {
    const kind = graphIR.kind as "forward" | "backward";
    sec(`Phase — Compiler optimisation: ${kind.toUpperCase()} graph`);

    const { graph: inputGraph } = importGraphIR(pkg, { kind });
    const graphValidation = validateGraph(inputGraph);

    if (!graphValidation.valid && logPasses) {
      console.error(`  ✗ ${kind} compiler graph validation failed:`);
      for (const e of graphValidation.errors) console.error(`    [${e.kind}] ${e.message}`);
      continue;
    } else if (logPasses) {
      console.log(`  ✓ ${kind} compiler graph valid — ${inputGraph.nodes.size} nodes`);
      printGraph(inputGraph, `${kind} input graph`);
    }

    const logs: Array<{ pass: string; level: string; message: string }> = [];
    const { pm, loopPass } = createDefaultPipeline({
      validateAfterEachPass: true,
      logSink: entry => {
        logs.push({ pass: entry.passName, level: entry.level, message: entry.message });
        if (logPasses) {
          const icon = entry.level === "error" ? "✗" : (entry.level === "warn" ? "⚠" : "·");
          console.log(`  [${entry.passName}] ${icon} ${entry.message}`);
        }
      },
    });

    const optimisedGraph = pm.run(inputGraph);

    if (logPasses) {
      const uniquePasses = [...new Set(logs.map(l => l.pass))].length;
      console.log(`\n  ✓ ${kind} pipeline complete. Passes run: ${uniquePasses}`);
      printGraph(optimisedGraph, `${kind} optimised graph`);
      printDiff(inputGraph, optimisedGraph, `${kind} full pipeline diff`);

      const loopModule = loopPass.getLastModule();
      if (loopModule) {
        printLoopModule(loopModule, `Loop IR (Optimised) — ${kind}`);
      }
    }
  }
}
