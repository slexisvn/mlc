// ─────────────────────────────────────────────────────────────────────────────
// GradRegistry — maps op names to GradBuilderFn implementations.
//
// Intentionally separate from OpSchemaRegistry so forward shape-inference
// (which belongs to core) and backward gradient logic (which belongs to
// autodiff) can be registered, queried, and updated independently.
// ─────────────────────────────────────────────────────────────────────────────

import { GradBuilderFn }       from "./types";
import { DEFAULT_GRAD_BUILDERS } from "./gradBuilders";

export class GradRegistry {
  private readonly _builders = new Map<string, GradBuilderFn>();

  constructor(builders: Record<string, GradBuilderFn> = {}) {
    for (const [op, fn] of Object.entries(builders)) {
      this._builders.set(op, fn);
    }
  }

  /** Register or overwrite a gradient builder for `op`. */
  register(op: string, builder: GradBuilderFn): this {
    this._builders.set(op, builder);
    return this;
  }

  has(op: string): boolean                     { return this._builders.has(op); }
  get(op: string): GradBuilderFn | undefined   { return this._builders.get(op); }
  getAll(): ReadonlyMap<string, GradBuilderFn> { return this._builders; }
}

/**
 * Pre-populated registry covering all built-in ops.
 * Import and pass to `buildBackwardGraph` or extend with `register()`.
 */
export const defaultGradRegistry = new GradRegistry(DEFAULT_GRAD_BUILDERS);
