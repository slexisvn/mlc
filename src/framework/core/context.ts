// ─────────────────────────────────────────────────────────────────────────────
//
// Active graph context — thread-local-style singleton that holds the currently
// active GraphBuilder and parameter accumulator during a session build.
//
// All nn.Module param allocation and functional ops resolve the active context
// via `getActiveBuilder()` rather than requiring it to be passed explicitly.
// ─────────────────────────────────────────────────────────────────────────────

import { GraphBuilder }  from "./graphBuilder";
import { ContextError }  from "./errors";
import { SymbolicTensor } from "../tensor/tensor";

// ─── ParamSpec ────────────────────────────────────────────────────────────────

/**
 * Associates a symbolic parameter tensor with its flat initial data.
 * Collected during session construction; serialised into IRPackage.parameters.
 */
export interface ParamSpec {
  readonly tensor:   SymbolicTensor;
  readonly data:     readonly number[];
  /** True when this tensor's value is fully known at compile time (constant folding). */
  readonly isConst?: boolean;
}

// ─── Active context state ─────────────────────────────────────────────────────

let _activeBuilder:   GraphBuilder | null  = null;
let _activeParamSink: ParamSpec[]  | null  = null;

// ─── Public accessors ─────────────────────────────────────────────────────────

/**
 * Return the currently active `GraphBuilder`.
 * @throws {ContextError} when called outside a session build.
 */
export function getActiveBuilder(): GraphBuilder {
  if (_activeBuilder === null) {
    throw new ContextError(
      "No active graph context.  Wrap your model construction in ExportSession.build().",
    );
  }
  return _activeBuilder;
}

/**
 * Return the currently active parameter sink (array to push ParamSpecs into).
 * @throws {ContextError} when called outside a session build.
 */
export function getActiveParamSink(): ParamSpec[] {
  if (_activeParamSink === null) {
    throw new ContextError(
      "No active param sink.  Wrap your model construction in ExportSession.build().",
    );
  }
  return _activeParamSink;
}

/** True when a session build is currently active. */
export function hasActiveContext(): boolean {
  return _activeBuilder !== null;
}

// ─── Context scope helper ─────────────────────────────────────────────────────

/**
 * Run `fn` with `gb` and `sink` set as the active context.
 * Restores the previous context (supports nested contexts).
 */
export function withActiveContext<T>(
  gb:   GraphBuilder,
  sink: ParamSpec[],
  fn:   () => T,
): T {
  const prevBuilder   = _activeBuilder;
  const prevParamSink = _activeParamSink;
  _activeBuilder   = gb;
  _activeParamSink = sink;
  try {
    return fn();
  } finally {
    _activeBuilder   = prevBuilder;
    _activeParamSink = prevParamSink;
  }
}
