import * as nn from "../framework/nn";
import { type Tensor, compile, zeros, tensor } from "../framework";

@compile({ autodiff: true, optimize: true, logPasses: true })
class Classifier extends nn.Module {
  private readonly fc1 = this.register("fc1", new nn.Linear(128, 256));
  private readonly fc2 = this.register("fc2", new nn.Linear(256, 256));
  private readonly fc3 = this.register("fc3", new nn.Linear(256, 128));
  private readonly fc4 = this.register("fc4", new nn.Linear(128, 64));
  private readonly fc5 = this.register("fc5", new nn.Linear(64, 32));

  forward(x: Tensor): Tensor {
    const h1 = this.fc1.forward(x).relu();

    const unused_h = this.fc2.forward(this.fc1.forward(x));
    const dead_tensor = unused_h.mul(tensor(5.0));

    const folded_const = tensor(3.0).add(tensor(2.0));

    const h2 = this.fc2.forward(h1).relu();

    const h2_scaled = h2.mul(folded_const);

    const expr_a = h2_scaled.sub(tensor(1.0));
    const expr_b = h2_scaled.sub(tensor(1.0));
    const combined = expr_a.add(expr_b);

    const fused_a = combined.add(tensor(0.5));
    const fused_b = fused_a.mul(tensor(0.5));

    const h3 = this.fc3.forward(fused_b).relu();
    const h4 = this.fc4.forward(h3).relu();

    return this.fc5.forward(h4).softmax();
  }
}

export function runPytorchCompilerDemo(): void {
  const SEP = "═".repeat(66);
  const sec = (title: string) => console.log(`\n${SEP}\n  ${title}\n${SEP}`);

  sec("Phase A — Model definition (PyTorch-like API)");

  const model = new Classifier();

  sec("Phase B — Trigger execution (JIT Trace & Compile)");
  console.log("  Instantiating an eager input tensor...");

  const x = zeros([128, 128], { dtype: "float32" });

  console.log("  Executing model.forward(x)...");
  const pkg: any = model.forward(x);

  if (pkg) {
    console.log(`\n  Demo complete. Total parameters exported: ${pkg.parameters?.length || 0}\n`);
  }
}

runPytorchCompilerDemo();