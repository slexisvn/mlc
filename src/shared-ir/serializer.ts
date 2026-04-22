// ─────────────────────────────────────────────────────────────────────────────
// shared-ir/serializer.ts
//
// JSON serialization and deserialization of IRPackage.
//
// Serializer responsibilities:
//   • Collect all GraphIR objects (forward + optional backward).
//   • Collect parameter data from a ParameterStore.
//   • Attach graph signatures for lightweight I/O inspection.
//   • Produce a JSON string (or parsed object) conforming to the IRPackage schema.
//
// Deserializer responsibilities:
//   • Parse and validate a JSON string or plain object.
//   • Return a typed IRPackage or throw a structured error on malformed input.
//
// Both functions are pure — they do not touch the file system.
// File I/O is the responsibility of the caller.
// ─────────────────────────────────────────────────────────────────────────────

import { IRPackage, GraphIR, GraphSignature, ParameterData } from "./schema";
import { validateIRPackage }                                  from "./validator";

// ─── Serializer ───────────────────────────────────────────────────────────────

export interface SerializeOptions {
  /** Include pretty-printed JSON output.  Defaults to false. */
  pretty?:           boolean;
  /** Op-set version string.  Defaults to "mini-ts-0.1". */
  opsetVersion?:     string;
  /** Arbitrary metadata to attach to the package. */
  metadata?:         Record<string, unknown>;
}

/**
 * Serialise one or more GraphIR objects (plus optional parameter data) into
 * an IRPackage JSON string.
 *
 * @param graphs      Ordered list of graph IRs (must contain at least one "forward" graph).
 * @param parameters  Optional list of serialised parameter values.
 * @param options     Serialisation options.
 * @returns JSON string of the IRPackage.
 */
export function serializeToJSON(
  graphs:     readonly GraphIR[],
  parameters: readonly ParameterData[] = [],
  options:    SerializeOptions = {},
): string {
  const opsetVersion = options.opsetVersion ?? "mini-ts-0.1";

  const signatures: GraphSignature[] = graphs.map(g => ({
    graphId: g.id,
    inputs:  g.inputIds.map(tid => {
      const t = g.tensors[tid];
      return { name: t.name, dtype: t.dtype, shape: [...t.shape] };
    }),
    outputs: g.outputIds.map(tid => {
      const t = g.tensors[tid];
      return { name: t.name, dtype: t.dtype, shape: [...t.shape] };
    }),
  }));

  const pkg: IRPackage = {
    irVersion:    "0.1",
    opsetVersion,
    graphs,
    parameters:   parameters.length > 0 ? parameters : undefined,
    signatures,
    metadata:     options.metadata,
  };

  return options.pretty
    ? JSON.stringify(pkg, null, 2)
    : JSON.stringify(pkg);
}

// ─── Deserializer ─────────────────────────────────────────────────────────────

export class DeserializationError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "DeserializationError";
  }
}

/**
 * Parse and validate an IRPackage from a JSON string or a plain object.
 *
 * @throws {DeserializationError} when the input is malformed JSON or fails
 *   structural validation.
 */
export function deserializeFromJSON(input: string | unknown): IRPackage {
  let raw: unknown;

  if (typeof input === "string") {
    try {
      raw = JSON.parse(input);
    } catch (e) {
      throw new DeserializationError("Failed to parse IRPackage JSON", e);
    }
  } else {
    raw = input;
  }

  // Basic type guard before running the full validator
  if (typeof raw !== "object" || raw === null) {
    throw new DeserializationError("IRPackage must be a JSON object");
  }

  const pkg = raw as IRPackage;
  const result = validateIRPackage(pkg);

  if (!result.valid) {
    const lines = result.errors
      .map(e => `  [${e.kind}]${e.graphId ? ` (graph "${e.graphId}")` : ""}: ${e.message}`)
      .join("\n");
    throw new DeserializationError(
      `IRPackage validation failed with ${result.errors.length} error(s):\n${lines}`,
    );
  }

  return pkg;
}
