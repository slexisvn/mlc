// ─────────────────────────────────────────────────────────────────────────────
//
// Structured error types for the framework frontend.
// ─────────────────────────────────────────────────────────────────────────────

export type FrameworkErrorKind =
  | "ShapeError"
  | "OpError"
  | "GraphBuildError"
  | "AutodiffError"
  | "ContextError";

export class FrameworkError extends Error {
  readonly kind: FrameworkErrorKind;
  constructor(kind: FrameworkErrorKind, message: string) {
    super(message);
    this.kind = kind;
    this.name = kind;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class ShapeError extends FrameworkError {
  constructor(message: string, readonly meta?: Record<string, unknown>) {
    super("ShapeError", message);
  }
}

export class OpError extends FrameworkError {
  constructor(message: string, readonly meta?: Record<string, unknown>) {
    super("OpError", message);
  }
}

export class GraphBuildError extends FrameworkError {
  constructor(message: string) {
    super("GraphBuildError", message);
  }
}

export class AutodiffError extends FrameworkError {
  constructor(message: string) {
    super("AutodiffError", message);
  }
}

export class ContextError extends FrameworkError {
  constructor(message: string) {
    super("ContextError", message);
  }
}
