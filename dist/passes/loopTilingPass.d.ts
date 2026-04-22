import { LoopModule } from "../ir/loopIR";
import { LoopPass, LoopPassResult } from "./loopPass";
export interface TilingConfig {
    /**
     * Tile size used when no per-variable override is present.
     * Default: 32.
     */
    readonly defaultTileSize: number;
    /**
     * Per-induction-variable tile size overrides.
     * Keys are variable names (e.g. "i", "j", "i0", "i1").
     * Values are the desired tile sizes.
     */
    readonly tileSizeByVar?: Readonly<Record<string, number>>;
    /**
     * Minimum static loop span required before tiling is applied.
     * Loops smaller than this are left untouched (avoids unnecessary overhead
     * for tiny dimensions like batch=1 or channels=3).
     * Default: 64.
     */
    readonly minBound: number;
    /**
     * Whether to tile reduction loops (loops where every direct Assign uses
     * accumulate=true, e.g. the matmul k-loop).
     * Default: false — reduction tiling requires careful handling of partial sums
     * and should be opt-in.
     */
    readonly tileReductions: boolean;
}
export declare const DEFAULT_TILING_CONFIG: TilingConfig;
export declare class LoopTilingPass implements LoopPass {
    readonly name = "LoopTilingPass";
    private readonly config;
    constructor(config?: Partial<TilingConfig>);
    run(module: LoopModule): LoopPassResult;
    /**
     * Tile a LoopStmt recursively.
     *
     * If the statement is an eligible ForLoop:
     *   • Strip-mine it into outer + inner loops.
     *   • Recurse into the inner loop's body (so inner dimensions are also tiled).
     * If not eligible:
     *   • Recurse into its body anyway (inner loops may still qualify).
     */
    private _tileStmt;
    private _shouldTile;
    private _getTileSize;
}
