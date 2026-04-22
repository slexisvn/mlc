// ─────────────────────────────────────────────────────────────────────────────
// framework/errors.ts
//
// Structured error types for the framework frontend.
//
// All errors extend FrameworkError so callers can use a single catch clause
// and then discriminate by `kind` for precise handling.
//
// Error taxonomy:
//   ShapeError      — shape mismatch or inference failure at graph-build time.
//   OpError         — unknown op, wrong arity, or unsupported attribute.
//   GraphBuildError — generic graph construction violation (dangling tensor, etc.).
//   AutodiffError   — gradient computation failure (non-differentiable op, etc.).
// ─────────────────────────────────────────────────────────────────────────────

export type FrameworkErrorKind =
  | "ShapeError"
  | "OpError"
  | "GraphBuildError"
  | "AutodiffError";

/**
 * Base class for all framework errors.
 *
 * `kind` is a discriminant for programmatic handling.
 * `message` is always human-readable.
 * `context` carries optional structured diagnostic data (op name, shapes, etc.).
 */
export class FrameworkError extends Error {
  readonly kind: FrameworkErrorKind;
  readonly context: Readonly<Record<string, unknown>>;

  constructor(
    kind: FrameworkErrorKind,
    message: string,
    context: Record<string, unknown> = {},
  ) {
    super(message);
    this.name    = kind;
    this.kind    = kind;
    this.context = context;
    // Preserve correct stack trace in V8
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}

// ─── Concrete error classes ───────────────────────────────────────────────────

/**
 * Thrown when tensor shapes are incompatible for an operation, or when
 * shape inference cannot determine the output shape.
 */
export class ShapeError extends FrameworkError {
  constructor(message: string, context: Record<string, unknown> = {}) {
    super("ShapeError", message, context);
  }
}

/**
 * Thrown when an op name is unregistered, the wrong number of arguments is
 * supplied, or an attribute value is invalid.
 */
export class OpError extends FrameworkError {
  constructor(message: string, context: Record<string, unknown> = {}) {
    super("OpError", message, context);
  }
}

/**
 * Thrown when the graph builder detects an invariant violation, such as a
 * tensor being used after the graph has been finalised, or a cycle being
 * introduced.
 */
export class GraphBuildError extends FrameworkError {
  constructor(message: string, context: Record<string, unknown> = {}) {
    super("GraphBuildError", message, context);
  }
}

/**
 * Thrown when the autodiff engine cannot differentiate through an op
 * (e.g. the op has no registered gradient builder, or an intermediate tensor
 * required for the backward pass has been pruned).
 */
export class AutodiffError extends FrameworkError {
  constructor(message: string, context: Record<string, unknown> = {}) {
    super("AutodiffError", message, context);
  }
}
