import {
  LoopModule,
  LoopFunction,
  LoopStmt,
  LoopExpr,
} from "../ir/loopIR";

const LINE = "─".repeat(62);
const THIN = "·".repeat(62);

export function printLoopModule(module: LoopModule, title?: string): void {
  console.log(`\n${LINE}`);
  if (title) console.log(`  ${title}`);
  const fnCount = module.functions.length;
  console.log(
    `  Loop IR Module : ${module.graphId}` +
    `  (${fnCount} function${fnCount !== 1 ? "s" : ""})`,
  );
  console.log(LINE);

  for (const fn of module.functions) {
    _printFunction(fn);
  }

  if (module.diagnostics.length > 0) {
    console.log(`  ▸ Diagnostics:`);
    for (const d of module.diagnostics) {
      console.log(`      ⚠  ${d}`);
    }
    console.log(THIN);
  }

  console.log(LINE);
}

function _printFunction(fn: LoopFunction): void {
  const inputs  = fn.params.filter(p => p.role === "input");
  const outputs = fn.params.filter(p => p.role === "output");
  const temps   = fn.params.filter(p => p.role === "temp");

  const fmtParam = (p: { name: string; shape: readonly number[] }): string =>
    `${p.name}(${p.shape.map(s => (s === -1 ? "?" : String(s))).join("×")})`;

  const inStr   = inputs.length  > 0 ? `in: ${inputs.map(fmtParam).join(", ")}`   : "in: —";
  const outStr  = outputs.length > 0 ? `out: ${outputs.map(fmtParam).join(", ")}`  : "out: —";
  const tempStr = temps.length   > 0 ? `  [temps: ${temps.map(fmtParam).join(", ")}]` : "";

  console.log(`  ▸ fn ${fn.name}  [${inStr}]  [${outStr}]${tempStr}`);

  if (fn.body.length === 0) {
    console.log(`    (empty body)`);
  } else {
    for (const stmt of fn.body) {
      _printStmt(stmt, 4);
    }
  }

  console.log(THIN);
}

function _printStmt(stmt: LoopStmt, indent: number): void {
  const pad = " ".repeat(indent);

  if (stmt.kind === "ForLoop") {
    const hiStr =
      stmt.hi === -1
        ? stmt.hiExpr !== undefined
          ? _fmtExpr(stmt.hiExpr)
          : "?"
        : String(stmt.hi);
    console.log(`${pad}for ${stmt.var.name} in [${stmt.lo}, ${hiStr}):`);
    for (const s of stmt.body) {
      _printStmt(s, indent + 2);
    }
  } else {
    const op  = stmt.accumulate ? "+=" : "=";
    console.log(`${pad}${_fmtExpr(stmt.target)} ${op} ${_fmtExpr(stmt.value)}`);
  }
}

function _fmtExpr(e: LoopExpr): string {
  switch (e.kind) {
    case "LoopVar":
      return e.name;

    case "Literal":
      if (e.isIndex) return String(Math.trunc(e.value));
      return Number.isInteger(e.value) ? `${e.value}.0` : String(e.value);

    case "MemRef":
      return `${e.buffer}[${e.indices.map(_fmtExpr).join(", ")}]`;

    case "BinOp":
      return `(${_fmtExpr(e.lhs)} ${e.op} ${_fmtExpr(e.rhs)})`;

    case "CallBuiltin":
      return `${e.callee}(${e.args.map(_fmtExpr).join(", ")})`;
  }
}
