// ─────────────────────────────────────────────────────────────────────────────
// frontend/nn/module.ts
//
// Module — base class for all neural network components.
//
// Key design decisions:
//   • No `GraphBuilder` or `ParameterStore` in the constructor.
//   • Parameter tensors are allocated lazily on the first `forward()` call
//     inside an active ExportSession context.
//   • Sub-module hierarchy via `register(name, child)` automatically derives
//     fully-qualified parameter names (e.g. "encoder.fc1.weight").
//   • `initParams()` is the extension point where subclasses declare their
//     trainable parameters using `this.addParam(...)`.
// ─────────────────────────────────────────────────────────────────────────────

import { getActiveBuilder, getActiveParamSink } from "../core/context";
import { SymbolicTensor }                        from "../tensor/tensor";
import { IRDType }                               from "../ir/schema";
import { ShapeExpr }                             from "../core/shape";
import { Initialiser, initXavier }               from "./parameter";

export abstract class Module {
  // ─── Hierarchy state ────────────────────────────────────────────────────
  private _localName:  string = "";
  private _parentPath: string = "";
  private _children:   Map<string, Module> = new Map();

  // ─── Param materialisation state ────────────────────────────────────────
  private _paramsReady: boolean = false;

  // ─── Path helpers ────────────────────────────────────────────────────────

  /** Fully-qualified path of this module (e.g. "encoder.fc1"). */
  get modulePath(): string {
    if (!this._localName) return this._parentPath;
    return this._parentPath ? `${this._parentPath}.${this._localName}` : this._localName;
  }

  /** @internal — called by parent's register() to propagate paths. */
  _setLocalName(name: string, parentPath: string): void {
    this._localName  = name;
    this._parentPath = parentPath;
    // Propagate updated path to all registered children
    for (const [childName, child] of this._children) {
      child._setLocalName(childName, this.modulePath);
    }
  }

  // ─── Sub-module registration ─────────────────────────────────────────────

  /**
   * Register a child module and return it.
   *
   * Call this in the constructor for every sub-module field:
   * ```ts
   * this.fc1 = this.register("fc1", new Linear(784, 256));
   * ```
   * This sets the fully-qualified parameter name prefix automatically.
   */
  protected register<T extends Module>(name: string, child: T): T {
    this._children.set(name, child);
    child._setLocalName(name, this.modulePath);
    return child;
  }

  // ─── Parameter allocation ─────────────────────────────────────────────────

  /**
   * Declare and allocate a trainable parameter tensor.
   *
   * Must be called from `initParams()`.  Requires an active graph context
   * (i.e. must be called within `ExportSession.build()`).
   */
  protected addParam(
    name:  string,
    shape: ShapeExpr,
    init:  Initialiser = initXavier,
    dtype: IRDType     = "float32",
  ): SymbolicTensor {
    const qualifiedName = this.modulePath ? `${this.modulePath}.${name}` : name;
    const gb            = getActiveBuilder();
    const sink          = getActiveParamSink();
    const tensor        = gb.param(qualifiedName, dtype, shape);
    sink.push({ tensor, data: init(shape) });
    return tensor;
  }

  // ─── Lazy initialisation ──────────────────────────────────────────────────

  /**
   * Override to declare trainable parameters using `this.addParam(...)`.
   * Called automatically on the first `forward()` invocation.
   *
   * Also triggers `_ensureParams()` on all registered children, so the entire
   * subtree is initialised in a single pass.
   */
  protected initParams(): void {
    // Default: no-op (modules with no parameters, e.g. ReLU, Sequential)
  }

  /** @internal — triggers param materialisation; called before forward. */
  protected _ensureParams(): void {
    if (this._paramsReady) return;
    this._paramsReady = true;
    this.initParams();
    for (const child of this._children.values()) {
      child._ensureParams();
    }
  }

  // ─── Abstract forward ─────────────────────────────────────────────────────

  abstract forward(...inputs: SymbolicTensor[]): SymbolicTensor | SymbolicTensor[];
}
