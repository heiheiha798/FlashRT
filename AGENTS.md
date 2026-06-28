# AGENTS.md instructions for /home/tianjianyang/code/FlashRT

<INSTRUCTIONS>
<!-- CODEGRAPH_START -->
## CodeGraph

This project has a CodeGraph MCP server (`codegraph_*` tools) configured. CodeGraph is a tree-sitter-parsed knowledge graph of every symbol, edge, and file. Reads are sub-millisecond and return structural information grep cannot.

### When to prefer codegraph over native search

Use codegraph for **structural** questions — what calls what, what would break, where is X defined, what is X's signature. Use native grep/read only for **literal text** queries (string contents, comments, log messages) or after you already have a specific file open.

| Question | Tool |
|---|---|
| "Where is X defined?" / "Find symbol named X" | `codegraph_search` |
| "What calls function Y?" | `codegraph_callers` |
| "What does Y call?" | `codegraph_callees` |
| "How does X reach/become Y? / trace the flow from X to Y" | `codegraph_trace` (one call = the whole path, incl. callback/React/JSX dynamic hops) |
| "What would break if I changed Z?" | `codegraph_impact` |
| "Show me Y's signature / source / docstring" | `codegraph_node` |
| "Give me focused context for a task/area" | `codegraph_context` |
| "See several related symbols' source at once" | `codegraph_explore` |
| "What files exist under path/" | `codegraph_files` |
| "Is the index healthy?" | `codegraph_status` |

### Rules of thumb

- **Answer directly — don't delegate exploration.** For "how does X work" / architecture questions, answer with 2-3 codegraph calls: `codegraph_context` first, then ONE `codegraph_explore` for the source of the symbols it surfaces. For a specific **flow** ("how does X reach Y") start with `codegraph_trace` from→to — one call returns the whole path with dynamic hops bridged — then ONE `codegraph_explore` for the bodies; don't rebuild the path with `codegraph_search` + `codegraph_callers`. Codegraph IS the pre-built index, so spawning a separate file-reading sub-task/agent — or running a grep + read loop — repeats work codegraph already did and costs more for the same answer.
- **Trust codegraph results.** They come from a full AST parse. Do NOT re-verify them with grep — that's slower, less accurate, and wastes context.
- **Don't grep first** when looking up a symbol by name. `codegraph_search` is faster and returns kind + location + signature in one call.
- **Don't chain `codegraph_search` + `codegraph_node`** when you just want context — `codegraph_context` is one call.
- **Don't loop `codegraph_node` over many symbols** — one `codegraph_explore` call returns several symbols' source grouped in a single capped call, while each separate node/Read call re-reads the whole context and costs far more.
- **Index lag — check the staleness banner, don't guess a wait.** When a codegraph response starts with "⚠️ Some files referenced below were edited since the last index sync…", the listed files are pending re-index — Read those specific files for accurate content. Files NOT in that banner are fresh and codegraph is authoritative for them. `codegraph_status` also lists pending files under "Pending sync".

### If `.codegraph/` doesn't exist

The MCP server returns "not initialized." Ask the user: *"I notice this project doesn't have CodeGraph initialized. Want me to run `codegraph init -i` to build the index?"*
<!-- CODEGRAPH_END -->

--- project-doc ---

任何场景下，如果开发过程中使用了兜底或者 fallback 策略，都先向我确认！！禁止擅自做主添加了兜底逻辑！！

只使用一次的功能禁止抽出一个独立helper！直接inline实现！

严禁直接使用 `git clean`、`git reset --hard`、`git checkout -- .`、`git restore .`
或其他等价命令清除所有未缓存/未跟踪文件。只有在我明确逐字要求清理这些文件时才允许执行。

在准备提交 PR 之前，必须先对照 `README.md` 中链接出去的 `CONTRIBUTING.md`
逐条检查本次改动是否满足对应要求，尤其是：
- PR 描述中的环境信息、命令、测试结果、precision / latency 证据
- kernel binding / CMake 改动的 build/import 验证要求
- 未覆盖项和限制必须明确写进 PR 描述

## Current Branch PR Runbook: Pi0.5 Thor Decoder FP4

本 branch 的工作目标是评估并实现 Pi0.5 action-expert decoder 在 Thor
SM110 上的 NVFP4/W4A4 小 M kernel。这里是本 branch 的单一事实来源，不再维护
`docs/todos/pi05-thor-decoder-fp4.md` 作为并行 todo。

### Scope

- 目标模型/路径：Pi0.5 Torch frontend on Thor SM110，decoder projections：
  `qkv`, `o`, `gate_up`, `down`，业务形状均为 `M=10`。
- 量化契约：NVFP4 activation + SFA，NVFP4 weight + SFB，输出保持 FP16。
- baseline：当前 production decoder FP8 path，即 cuBLASLt FP8 descale。
- 本地 4090/SM89 只用于编辑和静态检查；SM110 build、SASS、correctness、
  latency 结果必须以 Thor 为准。

### Thor Environment

- Thor host：`nvidia@10.7.229.14`
- Thor repo：`/home/nvidia/tianjianyang/FlashRT`
- Thor env：`/home/nvidia/tianjianyang/conda/envs/flashrt-thor`
- 远端命令优先使用：
  - `scripts/thor_ssh.sh '<remote command>'`
  - `bash scripts/run_thor_test.sh <cmd>` on Thor
- 本地同步可以用 `scripts/sync_dirty_to_thor.sh`；如果 SSH 需要密码，用
  `expect` 包一层，不要改成全树 delete/mirror 模式。

### Mandatory Benchmark Rules

Thor 是 Jetson-class 设备，未锁频时 GPU 会从约 315 MHz / 3 W 动态爬升，
此前所有 FP4/FP8 latency 噪声和 run-to-run 结论翻转都来自这里。

每次记录 latency 前必须先在 Thor 上执行：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

规则：

- 未锁频的 latency 只能作为 correctness smoke test，不能写进 PR 证据，也不能和
  FP8 比较。
- `jetson_clocks` 重启后失效，每次 reboot 后重锁。
- 新 kernel 的性能 gate 至少需要两次 same-session locked runs。
- 锁频 sanity check：`qkv` cuBLASLt FP8 descale 应约为 `6.2 us`。如果读到
  `12-17 us`，视为未锁频，停止并重锁。
- benchmark 必须用同一个 harness 同 session 比较候选 FP4、picked CUTLASS
  FP4、cuBLASLt FP8，避免跨 session 拼表。

### Hard Conclusions Already Established

- Generic CUTLASS FP4 is contract-correct but slower than cuBLASLt FP8 on the
  real Pi0.5 decoder projection shapes.
- cuBLASLt FP4 can run the FlashRT-equivalent layout with dense FP16 output,
  but current algorithms are not competitive for this workload.
- Hand-written scalar/SIMT and decode-to-FP16 WMMA/PTX variants are rejected as
  performance paths. They are diagnostic only. Locked-clock Thor evidence:
  `smallm_fp4_cuda` is approximately `39/31/116/56 us` for
  `qkv/o/gate_up/down`, and `wmma_fp4_cuda` is even slower
  (`134/190/389/373 us`). cuBLASLt FP8 is approximately
  `6.2/6.2/10.7/8.8 us` in the same session.
- Triton inline decode validated packed FP4/SFA/SFB indexing, but generated
  FP4 decode plus FP16 `mma.sync`, not native FP4 tensor-core codegen. It is not
  a performance candidate.
- Transposed CUTLASS `W[N,K] @ A_pad[16,K]^T -> C_t[N,16]` with v14/v15 is
  correct and emits native `UTCOMMA.4X`, but locked-clock four-shape evidence
  shows it only reaches FP8 parity and `o` loses about 5%。这条路线不是
  PR-worthy runtime path。
- Transposed-direct CUTLASS `W[N,K] @ A[M,K]^T -> T[N,M]` with ColumnMajor D
  stride writes directly into row-major `D[M,N]` and is correct, but locked
  evidence is still parity: best ratios vs cuBLASLt FP8 are about
  `1.01x / 0.98-1.05x / 1.00x / 1.00x` for `qkv/o/gate_up/down`. Removing the
  temporary transposed output is not the missing performance lever.
- Explicit CUTLASS schedule selection on the same transposed/direct boundary
  is also not the missing lever. A temporary v16-v19 experiment forced
  `KernelTmaWarpSpecialized1SmNvf4Sm100` and generic
  `KernelTmaWarpSpecialized1SmBlockScaledSm100` for `tile128x16x{128,256}`.
  It compiled and was correct on Thor, but locked `down M=10,N=1024,K=4096`
  stayed at parity: best FP4 was `cutlass_fp4_transposed_direct_v15 8.214 us`
  versus cuBLASLt FP8 descale `8.213 us`. The explicit schedule instances were
  removed from the code; do not re-add more CUTLASS schedule variants unless a
  new structural hypothesis changes data movement.
- 2CTA native FP4 atom was tested as a data-movement/W-streaming hypothesis,
  not a schedule-name sweep. The low-level atom probe compiles to
  `UTCOMMA.2CTA.4X`, and the CUTLASS builder accepts
  `TileShape<256,16,{128,256}>, Cluster<2,1,1>` on Thor, but locked `down`
  benchmark still sits at parity: `tile256x16x256` direct was `8.215 us`,
  best FP4 remained `tile128x16x256` transposed at `8.207 us`, and cuBLASLt
  FP8 descale was `8.206 us`. The temporary v16/v17 instances should not be
  kept as runtime candidates.
- CUTLASS example 92 exposes a separate mixed TMA+CPAsync block-scaled FP4
  mainloop for decoding/MoE-style small token counts. This is a valid new
  data-movement hypothesis if W is mapped to logical A/TMA and the tiny
  activation side is mapped to logical B/CPAsync. It is not the same as the
  already rejected Auto/TMA schedule sweep. The benchmark-only transposed-direct
  variants v16/v17 compile and are correct on Thor, but locked `down`
  warmup 30 / iters 200 loses: v16 `10.693 us`, v17 `11.317 us`,
  transposed-direct v15 `8.218 us`, cuBLASLt FP8 descale `8.751 us`. This
  hypothesis is rejected as a performance direction.
- Latest locked four-shape check, same session, warmup 30 / iters 200, confirms
  the best CUTLASS FP4 path is still parity rather than a stable runtime win:
  `qkv` best FP4 `6.310 us` vs FP8 `6.191 us` (`1.019x` slower),
  `o` `6.190 us` vs `6.217 us` (`0.996x`),
  `gate_up` `10.267 us` vs `10.276 us` (`0.999x`), and
  `down` `8.215 us` vs `8.211 us` (`1.000x`). A single-shape `down` rerun once
  showed `0.939x`, but the full same-session run did not reproduce a stable
  all-shape gain. Do not claim runtime improvement from current CUTLASS
  variants.
- The current transposed-direct grid has very low CTA count on some shapes
  after mapping W to logical M and activation rows to logical N. For example,
  `down` is only `ceil(1024/128) * ceil(10/16) = 8` output tiles before any
  scheduler decomposition. This points to K-axis parallelism / StreamK, not
  another local tile wrapper, as the next structural hypothesis. The repo
  already has a similar SM120 precedent:
  `csrc/gemm/fp4/cutlass_nvfp4_gemm_dn_streamk_bias_sm120.cu` recovered
  utilization for a low-CTA down GEMM by using `cutlass::gemm::StreamKScheduler`.
- StreamK/SplitK was tested as a benchmark-only transposed-direct diagnostic
  after the low-CTA hypothesis above. It is correct, but rejected on locked
  Thor `down M=10,N=1024,K=4096`, warmup 30 / iters 200:
  baseline transposed-direct v15 `8.224 us`, cuBLASLt FP8 descale `8.203 us`,
  StreamK v18 split2/4/8/16 = `10.024/12.340/16.998/26.331 us`,
  StreamK v19 split2/4/8/16 = `10.141/12.589/17.473/26.039 us`.
  The reduction/scheduler overhead dominates any extra K-axis parallelism.
- CUTLASS example 91 `GemvBlockScaled` can be adapted to this contract after
  fixing shared-W batching and W scale layout, but locked-clock evidence shows
  it is far slower than FP8. It is rejected as a runtime direction.
- The CUTLASS small-N guard patch is an explicit research prerequisite only.
  Do not present it as generic SM100 support.

### SM110 Naming And CUTLASS Boundary

- CUTLASS uses `Sm100`, `MainloopSm100...`, `SM100_MMA_MXF4...` names for this
  Blackwell/Thor UMMA-family path. That does not mean the FlashRT feature should
  be described as generic SM100 support.
- FlashRT-facing code and PR text must say Thor/SM110 or `sm_110a` unless we
  have non-Thor validation.
- v10-v15 variant exposure is gated by `FLASHRT_ENABLE_SM110_SMALL_N_FP4`, which
  CMake defines only when `GPU_ARCH STREQUAL "110"`.
- `patches/cutlass/sm110-blockscaled-small-n.patch` and
  `scripts/apply_cutlass_sm110_small_n_patch.sh` are research tools to reproduce
  `CTA_N=16` builder experiments, not a final PR shape.

### Current Direction

Claude's latest session established the better direction, and the GEMV
diagnostic has now been completed:

```text
FlashRT-owned Thor/SM110 small-M FP4 kernel path.
Rejected candidate: CUTLASS GemvBlockScaled diagnostic for Pi0.5 decoder.
Rejected candidate: CUTLASS mixed TMA+CPAsync transposed-direct diagnostic.
Current candidate: custom CuTe/native-FP4 kernel for down, or a larger decoder
fusion boundary that amortizes the 128-row FP4 atom and generic epilogue cost.
```

Why this direction is justified:

- Decoder `M=10` makes A/D traffic small; W traffic dominates.
- For `down`, FP4 reads roughly `0.56x` the W bytes of FP8, but current
  transposed CUTLASS FP4 only ties FP8.
- The current transposed-direct path already moves the 128-row SM100 FP4 atom
  onto the weight-N dimension (`logical_M = original N`), so the remaining
  structural waste is not the original generic-GEMM `M=10 -> tile_M=128` floor.
  The residual waste is business `M=10` mapping to atom/tile `N=16`, plus
  generic CUTLASS mainloop/epilogue/scheduler work that was built for broader
  GEMM shapes rather than this fixed decoder boundary.
- Locked profiling/latency evidence says FP4 mainloop SM throughput is only
  `3-6%`; implied effective W bandwidth is roughly `307 GB/s` for FP4 versus
  `~550 GB/s` for FP8 on the same device/session. The absolute bandwidth number
  is approximate, but the ratio shows FP4 leaves substantial bandwidth unused.
- Therefore the next useful work is memory-throughput-oriented small-M scheduling,
  not more Python routing or generic CUTLASS tile sweeps.

### Active Implementation State

- `csrc/gemm/fp4/cutlass_fp4_gemv.{cu,cuh}` is a rejected diagnostic candidate.
  It adapts CUTLASS example 91 `GemvBlockScaled`:
  `D[M,N] = A[M,K] @ W[N,K]^T`, with W as the GEMV matrix and each A row as a
  batch vector.
- The GEMV path includes an SFA-to-GEMV-SFB repack helper. Correct mapping
  requires `batch_stride_A=0` because W is shared across activation rows, and W
  scales must use matrix-A SFA layout replicated once per activation-row batch.
- Thor locked-clock two-run result, warmup 30 / iters 200:
  `gemv_fp4_cutlass` is approximately `41/28/123/45 us` for
  `qkv/o/gate_up/down`, while cuBLASLt FP8 is approximately
  `6.2/6.2/10.3/8.3 us`. This is not close enough to polish.
- `tests/bench_pi05_decoder_fp4_primitives.py` is the current unified primitive
  benchmark harness. It uses CUDA Graph timing and prints correctness plus
  same-session latency for candidate FP4, picked CUTLASS FP4, cuBLASLt FP4, and
  FP8 baselines.
- If `gemv_fp4_cutlass` is present in `results`, it must be included in the
  `fp4_keys` best-FP4 summary. Do not let summary metrics omit a new candidate.
- `cutlass_fp4_gemm_transposed_direct_variant` is a rejected diagnostic entry.
  It is useful evidence that direct row-major writeback does not turn native
  CUTLASS FP4 into a stable win, but it should not be the final PR runtime path.
- `csrc/gemm/fp4/pi05_decoder_fp4_smallm.{cu,cuh}` is a rejected diagnostic
  scalar/SIMT + WMMA prototype. It is useful as a correctness sanity row but
  must not be presented as the final PR kernel.

### Immediate Next Steps

1. Keep `tests/bench_pi05_decoder_fp4_primitives.py` as the shared evidence
   harness for rejected and active primitive candidates.
2. Start the real custom kernel as a Thor/SM110-specific native-FP4 path. First
   target: `down`, `M=10,N=1024,K=4096`.
   - The kernel must change scheduling/data movement, not just output stride.
   - It must avoid decode-to-FP16 + HMMA; native FP4 tensor-core evidence is
     required before claiming FP4 acceleration.
   - The target is memory throughput: W bytes dominate at `M=10`.
   - The remaining credible CUTLASS data-movement boundary, mixed
     TMA+CPAsync transposed-direct with W as TMA-loaded logical A and activation
     as CPAsync-loaded logical B, has now been tested and rejected. Do not add
     more CUTLASS wrapper variants unless a new structural dataflow hypothesis
     is written down first.
   - StreamK/SplitK transposed-direct was tested and rejected. Further CUTLASS
     scheduler variants are unlikely to be useful unless they remove reduction
     overhead or fuse a larger decoder boundary.
3. Do not change frontend runtime routing, public flags, or `pick_variant`
   defaults until the kernel-level evidence beats cuBLASLt FP8 on all four
   projections.

### Native Kernel Requirements

- A serious FP4 kernel must emit native FP4 tensor-core codegen:
  `UTCOMMA` / `tcgen05` / `SM100_MMA_MXF4`-style instructions.
- A kernel that only does FP4 decode to FP16 plus `HMMA` / `mma.sync` is
  diagnostic only, regardless of correctness.
- First custom target should be `down`: `M=10,N=1024,K=4096`.
- Use picked CUTLASS FP4 output as correctness reference.
- Do not change frontend runtime routing, public flags, or `pick_variant`
  defaults until kernel-level evidence is positive.

### PR Gates

Do not submit a PR for this branch until all applicable gates are met:

- Thor `sm_110a` build/import succeeds after a clean sync.
- Correctness is reported against existing CUTLASS FP4 for all covered shapes.
- Latency is from locked-clock same-session runs.
- Candidate FP4 is faster than cuBLASLt FP8 on all four real decoder projections
  if the PR changes runtime behavior.
- SASS evidence confirms native FP4 when claiming tensor-core FP4.
- If CMake/pybind/kernel bindings changed, include build/import commands and
  results in the PR description per `CONTRIBUTING.md`.
- Explicitly state limitations: Thor-only validation, benchmark-only path, or
  any deferred projection/runtime integration.

</INSTRUCTIONS>
