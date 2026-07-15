# Jetson-PI Provider Validation Matrix

This document is the reviewable record for `jetsonpi迁移.txt` §14. It
separates correctness gates from diagnostic performance measurements. Timings
are single-run wall-clock diagnostics from the named logs, not statistically
stable benchmarks and not cross-backend speed claims.

## Validated Revisions

- FlashRT validation baseline:
  branch `jetson-pi-link-check` (fork `heiheiha798/FlashRT`), tracked by
  https://github.com/flashrt-project/FlashRT/issues/143. The provider
  implementation was complete at `0080e3daa383b077955d3063f2ab755de1996dca`;
  later commits through this baseline corrected and finalized the validation
  record. (The pre-rebase history recorded baseline `91a5cc3`; commit SHAs
  shift on rebase, so the branch + issue are the stable reference.)
- Jetson-PI revision:
  public repository https://github.com/PKU-SEC-Lab/Jetson-PI-Edge, branch
  `Jetson-PI-flashrt`, tip `5de3f9e210086f7bc04c5a434990bd28e7ed2240`
  (2026-07-15). Originally developed against a private fork (branch
  `flashrt-migration-merge`, commit
  `9c8e8b30e629a2475958c7b8a0bfb06f6f05295d`, based on
  `origin/merge@436fdb2aceaf564e152be6e0779180f56a279074`; the prior migration
  is preserved at `origin/flashrt-migration-master-baseline@68dd395b3f89dbd031ae564e335780f702fbd1e7`).
  That history has since been opened as the public `Jetson-PI-flashrt` branch,
  so the private SHAs are kept here only as provenance.
- CUDA/Vulkan/SYCL device used for final accelerator runs: physical GPU 6,
  NVIDIA GeForce RTX 4090. CUDA used `CUDA_VISIBLE_DEVICES=6`; Vulkan used
  `GGML_VK_VISIBLE_DEVICES=6`; SYCL used `ONEAPI_DEVICE_SELECTOR=cuda:6` with
  the matching Unified Runtime CUDA adapter.

## Migration Requirement Coverage

| Source requirement | Delivered result | Review gate |
|---|---|---|
| Phase 0: baseline and contracts | Locked Jetson-PI/GGML revision and build identity; full-file model/mmproj SHA-256; explicit port shape/dtype/ownership; native Pi0 CLI input, action output, and timing baseline. | CLI and FlashRT use the same two RGB fixtures, 32-F32 state, effective prompt, CUDA backend, and fixed-seed policy; all 320 action elements are bit-identical. |
| Phase 1: in-process Pi0 whole graph | `frt_model_runtime_v2` callback `infer` stage with staged images/prompt/state/actions; provider owns Jetson-PI/MTMD/GGML state and exposes no GGML type in FlashRT ABI. The migrated narrow API detects and pins Pi0 versus Pi0.5 per handle. | Pi0 and Pi0.5 CPU/CUDA/SYCL clean-exit model runs, Vulkan inference-success/teardown-fail runs, and direct narrow-C-API parity. |
| Phase 2: Python and C API | One C++ implementation is reached through the FlashRT C ABI, C++ factory, and ctypes `flash_rt.load_model(framework="jetson_pi")`; lifecycle, errors, stale output, and multiple instances are tested. | Native C++ and Python smoke suites, including installed-provider loading. |
| Phase 3: generic text LLM | Prompt/optional tokens, next-token, EOG, full logits, accumulated text, `reset -> prefill -> decode`, private KV/sampler state, host interruption, and hard token budget. | Qwen3 0.6B, TinyLlama 1.1B, and Gemma 2 2B on CPU/CUDA; Qwen3 on clean-exit SYCL and inference-success/teardown-fail Vulkan; direct parity and interleaved-session isolation. |
| Phase 4: multimodal LLM | Images + prompt, provider-private MTMD/VIT embeddings, one-shot and repeatable prefill/decode, next-token/EOG/logits/text outputs. | Qwen3-VL 2B CPU/CUDA/SYCL exact direct parity; Vulkan inference reaches success before the recorded NVIDIA ICD teardown fault. |
| Phase 5: finer Pi0 stages | Stable narrow `context`/`action` C API and FlashRT `context -> action` DAG; pending context is single-use; encoded cross-KV and strictly guarded decoder graph reuse remain provider-private. | Whole/split/direct parity is exact; stale/replaced/failed input invalidation is tested; graph profiling confirms one rebuild plus nine reuses. |
| Phase 6: backend and zero-copy evolution | Explicit CPU/CUDA/Vulkan/OpenCL/SYCL selection; opaque memory-domain token; host map/unmap; read-only versioned DLPack; dynamic GGML backend package. | Model-by-backend matrix below; OpenCL is an explicit capability gate; pointer identity, mapping lifetime, module loading, and hard-failure paths are tested. |
| §5.2/§15 production delivery | Dev `add_subdirectory` and installed `find_package(JetsonPI)` routes, exact-prefix RPATH, dynamic backend module validation, locked fork, and narrow-library symbol audit. | Standalone install, out-of-tree FlashRT consumer, `readelf`/`nm` audits, C++ CPU/CUDA, and installed `_c.so` Python CUDA runs. |
| §10.2 and §12 recommendations | A full generic `exec/backend` vtable and the illustrative directory reshuffle are not required for the provider contract. The implemented minimal memory-domain contract preserves the existing CUDA Graph path without moving verified code for style alone. | Design decision and re-entry criteria are recorded in `phase6_backend_vtable_eval.md`; no functional requirement depends on the suggested file layout. |
| §14 correctness/performance matrix | Shape/value/finite checks, repeatability, context/KV isolation, port-schema identity, checkpoint/backend/stage fingerprinting, latency breakdowns, token/s, and sampled accelerator memory. | Matrix, invariants, logs, and reproduction commands below. |

## Correctness Matrix

| Model face | Model | Backend | Result | Evidence |
|---|---|---|---|---|
| Pi0 | Pi0 Base F16 + VIT mmproj | CPU | PASS | FlashRT versus direct `jetson_pi_pi0` action parity, `max_abs_diff=0`; shape `(10,32)`; finite actions. |
| Pi0 | Pi0 Base F16 + VIT mmproj | CUDA | PASS | 37/37 model layers and VIT on CUDA; whole infer repeatability; no context leak; FlashRT/direct parity `max_abs_diff=0`; whole versus context/action and native CLI versus FlashRT are bit-identical. |
| Pi0 | Pi0 Base F16 + VIT mmproj | Vulkan | INFERENCE/PARITY PASS; TEARDOWN FAIL | 37/37 model layers plus VIT on Vulkan; FlashRT/direct parity `max_abs_diff=0` reaches `== PI0 PARITY PASSED ==`, then process exit is 139. |
| Pi0 | Pi0 Base F16 + VIT mmproj | SYCL-on-CUDA | PASS | 37/37 model layers plus VIT on SYCL0; whole/direct and whole/context-action parity `max_abs_diff=0`; repeated inference and host-SWAP/DLPack gates pass. |
| Pi0.5 | Pi05 LIBERO F16 (371 tensors) + VIT mmproj | CPU | PASS | Automatic Pi0.5 detection selects the Pi0.5 text/VIT adapters; finite `(10,32)` actions; whole/split/direct `max_abs_diff=0`; host cross-KV is refreshed on every denoise step. |
| Pi0.5 | Pi05 LIBERO F16 (371 tensors) + VIT mmproj | CUDA | PASS | 37/37 model layers plus VIT on CUDA; all 371 model tensors load; whole/split/direct `max_abs_diff=0`; installed Python host-SWAP and read-only DLPack gates pass. |
| Pi0.5 | Pi05 LIBERO F16 (371 tensors) + VIT mmproj | Vulkan | INFERENCE/PARITY PASS; TEARDOWN FAIL | 37/37 model layers plus VIT on Vulkan; whole/split/direct `max_abs_diff=0` reaches `== PI0 PARITY PASSED ==`, then process exit is 139. |
| Pi0.5 | Pi05 LIBERO F16 (371 tensors) + VIT mmproj | SYCL-on-CUDA | PASS | 37/37 model layers plus VIT on SYCL0; whole/split/direct `max_abs_diff=0`; finite actions and clean exit. |
| Text LLM | Qwen3 0.6B Q4_K_M | CPU | PASS | FlashRT/direct greedy text, first token, and complete prefill logits parity; logits `max_abs_diff=0`. |
| Text LLM | Qwen3 0.6B Q4_K_M | CUDA | PASS | 29/29 layers offloaded; FlashRT/direct text, first token, and complete prefill logits parity with `max_abs_diff=0`; two distinct interleaved sessions reproduce standalone token/EOG sequences. |
| Text LLM | Qwen3 0.6B Q4_K_M | Vulkan | INFERENCE/PARITY PASS; TEARDOWN FAIL | 29/29 layers offloaded; FlashRT/direct greedy text exact parity is retained, and the final rebuilt provider test reaches `== JETSON_PI LLM PASSED ==`, then process exit is 139. |
| Text LLM | Qwen3 0.6B Q4_K_M | SYCL-on-CUDA | PASS | 29/29 layers on SYCL0; staged decode, interleaved session isolation, logits, interruption, and token-budget gates pass with clean exit. |
| Text LLM | TinyLlama 1.1B | CPU/CUDA | PASS | 23/23 CPU and 23/23 CUDA model matrix runs complete. |
| Text LLM | Gemma 2 2B | CPU/CUDA | PASS | 27/27 CPU and 27/27 CUDA model matrix runs complete. |
| MLLM | Qwen3-VL 2B Q4_K_M + F16 mmproj | CPU | PASS | FlashRT and direct `jetson_pi_mllm` produce exact image+text output for the same generated RGB fixture and greedy sampling. |
| MLLM | Qwen3-VL 2B Q4_K_M + F16 mmproj | CUDA | PASS | 29/29 language layers plus CLIP/VIT on CUDA; FlashRT/direct image+text output is exact; one-shot equals staged prefill/decode; finite logits. |
| MLLM | Qwen3-VL 2B Q4_K_M + F16 mmproj | Vulkan | INFERENCE PASS; TEARDOWN FAIL | 29/29 language layers plus CLIP/VIT on Vulkan; one-shot/staged, finite-logit, and budget gates reach `== JETSON_PI MLLM PASSED ==`, then process exit is 139. |
| MLLM | Qwen3-VL 2B Q4_K_M + F16 mmproj | SYCL-on-CUDA | PASS | 29/29 language layers plus CLIP/VIT on SYCL0; FlashRT/direct output exact; one-shot/staged decode, finite logits, and token budget pass. |
| LLM provider | Qwen3 0.6B | OpenCL | BACKEND CAPABILITY GATE | NVIDIA OpenCL 3.0 platform/device is enumerated, but this fork accepts only Adreno/Qualcomm and Intel families, drops the RTX 4090, and explicit open fails with zero registered devices. |
| Production package | Qwen3 0.6B plus installed Python Pi0/Pi0.5/MLLM | direct-link and `GGML_BACKEND_DL` CPU/CUDA | PASS | Direct-link dependencies resolve to the locked install prefix and CUDA reports 29/29; dynamic-package core/provider ELF has no backend-module `DT_NEEDED`, CPU reports 0/29, CUDA reports 29/29, and C++ plus installed Python gates pass. The installed MTMD header closure includes `pi-model.h`, and out-of-tree static/shared consumers build without source-tree headers. |

## Known Driver-Lifecycle Result

- With the final rebuilt binaries, Pi0, Pi0.5, Qwen3 0.6B, and Qwen3-VL 2B all reach
  their test success markers after running the intended Vulkan model/VIT paths.
  Each process then segfaults during NVIDIA 550.54.14 driver teardown and exits
  139. No Vulkan row is counted as a clean PASS, and the provider does not
  install a fallback or speculative teardown workaround.

## §14 Invariants

- Consecutive Pi0 requests invalidate stale actions and reproduce the same
  fixed-seed output for identical inputs; no previous context leaks.
- CPU Pi0/Pi0.5 graph reuse refreshes host-resident encoded cross-KV on every
  denoise step. The GPU-resident path retains its explicit ready guard; the
  host path no longer reuses stale graph-input storage and produces finite,
  exact-parity actions.
- A failed replacement of any known Pi0/LLM/MLLM input port clears that port's
  ready state before returning. A later stage therefore hard-fails instead of
  reusing the previous request's value. RGB conversion rejects negative or
  undersized row strides and verifies the complete last-row byte span.
- Pi0 actions are checked for shape, nonzero values, and NaN/Inf absence.
- Pi0 host SWAP is consumed zero-copy through NumPy's versioned, read-only
  DLPack path; pointer identity and the live-consumer stage guard are tested.
  PyTorch 2.6 legacy DLPack export is rejected because it cannot preserve the
  read-only provider mapping, and no implicit copy fallback is used.
- Pi0 token `copy_to_host` rejects a null destination for nonzero copies.
- LLM/MLLM one-shot size-query uses `max_tokens` times the largest serialized
  token piece in the loaded vocabulary, not a fixed bytes-per-token guess.
  FlashRT and its Python frontends then query and read the actual completed
  text byte count from the staged output port.
- The narrow LLM API rejects prompt lengths that cannot be represented by the
  llama tokenizer's `int32_t` length. Pi0 and both MLLM entry points reject
  zero or overflowing RGB dimensions before bitmap construction; staged MLLM
  `prefill` enforces the same boundary as one-shot inference.
- LLM/MLLM prefill exposes finite vocabulary logits; decode is host-repeatable.
- Host interruption is explicit: callers stop generation by not issuing another
  decode stage. Each staged session enforces its configured `max_tokens`; one
  extra decode hard-fails. The narrow API restores the budget on `prefill`,
  while FlashRT conventionally starts sessions with reset+prefill.
- Same-session KV is reused across decode steps; distinct LLM sessions are
  tested with different prompts and compared token-by-token with standalone
  baselines.
- Backend selection is an explicit string and unavailable/unknown backends
  hard-fail. A provider unit test verifies that changing backend changes the
  fingerprint while preserving the complete port schema.
- OpenCL validation sets `OCL_ICD_VENDORS=/etc/OpenCL/vendors` so the NVIDIA
  OpenCL 3.0 ICD is actually enumerated. `ggml-opencl` then rejects the NVIDIA
  family, registers zero devices, and the requested OpenCL provider hard-fails.
- A dynamic GGML package validates every backend module declared by its own
  `ggml-config.cmake`; runtime modules are not linked into the provider. Missing,
  relative, or cross-prefix backend directories fail production configuration.
- Pure-CPU production validation hides CUDA devices with
  `CUDA_VISIBLE_DEVICES=`. Otherwise llama.cpp creates contexts for visible CUDA
  devices even when model weights report `offloaded 0/29`; that is not a pure
  CPU execution claim and is not used as evidence here.
- Standard JSON open computes full-file SHA-256 for model and mmproj files.
  A same-size edit after byte 64 KiB changes the fingerprint. Missing or
  unreadable checkpoints fail before factory creation and expose a thread-local
  open error; there is no path-only fallback.
- Canonical runtime identity also fingerprints quantization-bearing checkpoint
  bytes, backend, port schema, callback-stage DAG, and executable layout.

## Diagnostic Performance Snapshot

These values come from existing single-run logs and show that every requested
measurement point is observable. They are not acceptance thresholds.

| Model/backend | Diagnostic values |
|---|---|
| Pi0 CPU | VIT about 6.4-6.6 s per view, encode about 2.17 s, ten-step action decode about 39.38 s. |
| Pi0 CUDA | Warm VIT about 7.6-9.8 ms per view, encode about 102-114 ms, ten-step action decode about 1.10-1.52 s. Graph reuse triggers one rebuild plus nine reuses; build+alloc is below 1% of step time, so compute remains dominant. |
| Pi0 native CLI CUDA | Exact-parity fixture: two image slices 19/8 ms, VIT diagnostic 19.09 ms, encode 481.17 ms, ten-step action decode 117.08 ms, context total 969.96 ms, and process wall time 4.51 s including model/mmproj load and built-in warmup. |
| Pi0 SYCL-on-CUDA | Warm VIT 25.82/27.83 ms for two views, encode 189.22 ms, and ten-step action decode 215.52 ms; the first cold tick was 209.74/38.59 ms, 497.99 ms, and 496.37 ms respectively. |
| Qwen3 0.6B CPU | Dynamic-package staged prefill 4365.797 ms; 16-token repeatable decode 0.25 token/s (63784.955 ms). |
| Qwen3 0.6B CUDA | Staged prefill 5.909 ms; repeatable decode 433.98 token/s (16 tokens in 36.868 ms). The independently installed dynamic package measured 6.076 ms and 429.75 token/s. |
| Qwen3 0.6B SYCL-on-CUDA | Staged prefill 12.802 ms; repeatable decode 149.02 token/s (16 tokens in 107.368 ms). |
| TinyLlama 1.1B CPU/CUDA | 12-token batch diagnostics about 2.66-2.79 s on CPU and 4.1-30.6 ms on CUDA. |
| Gemma 2 2B CPU/CUDA | 12-13-token batch diagnostics about 3.92-7.79 s on CPU and 6.5-35.2 ms on CUDA. |
| Qwen3-VL 2B CPU | Warm VIT 315 ms, staged multimodal prefill 13248.101 ms, and repeatable decode 0.25 token/s (16 tokens in 64684.120 ms). |
| Qwen3-VL 2B CUDA | Warm VIT 5 ms, staged multimodal prefill 22.881 ms, and repeatable decode 295.27 token/s (16 tokens in 54.188 ms). |
| Qwen3-VL 2B SYCL-on-CUDA | Warm VIT 12-16 ms, staged multimodal prefill 61.547 ms, and repeatable decode 152.04 token/s (16 tokens in 105.233 ms). |
| Qwen3-VL 2B Vulkan | 58-59-token multimodal prompt batches about 39-98 ms before the known process-exit driver teardown failure. |

At 50 ms `nvidia-smi` sampling on physical GPU6, the complete Qwen3 LLM CUDA
test peaked at 2914 MiB and the SYCL-on-CUDA test at 2844 MiB from a 0 MiB
baseline; both returned to 0 MiB after clean exit. These are process-level
observations for the full multi-session test, not allocator-internal counters.

Final-run logs used by this matrix include:

- `/tmp/merge-pi0-parity-cpu-fixed.log`
- `/tmp/merge-pi0-parity-cuda-fixed.log`
- `/tmp/merge-pi0-parity-sycl-gpu6.log`
- `/tmp/merge-pi0-parity-vulkan-gpu6.log`
- `/tmp/merge-pi05-parity-cpu.log`
- `/tmp/merge-pi05-parity-cuda-fixed.log`
- `/tmp/merge-pi05-parity-sycl-gpu6.log`
- `/tmp/merge-pi05-parity-vulkan-gpu6.log`
- `/tmp/merge-llm-parity-{cpu,sycl-gpu6,vulkan-gpu6}.log`
- `/tmp/merge-llm-parity-cuda-fixed.log`
- `/tmp/merge-mllm-parity-{cpu,sycl-gpu6,vulkan-gpu6}.log`
- `/tmp/merge-mllm-parity-cuda-fixed.log`
- `/tmp/merge-llm-opencl-nvidia-gate.log`
- `/tmp/merge-sycl-upscale-backend-ops-gpu6.log`
- `/tmp/merge-prod-direct-llm-cuda-gpu6.log`
- `/tmp/merge-prod-dynamic-llm-{cpu,cuda-gpu6}.log`
- `/tmp/merge-installed-python-{llm,pi0,pi05,mllm}-cuda-gpu6.log`
- `/tmp/merge-reviewfix-pi05-parity-cuda-gpu6.log`
- `/tmp/merge-reviewfix-mllm-parity-cuda-gpu6.log`

The following older logs remain the historical pre-`origin/merge` evidence:

- `/tmp/llm-token-logit-parity-cpu-final.log`
- `/tmp/llm-token-logit-parity-cuda-final-gpu6.log`
- `/tmp/mllm-parity-cpu-final.log`
- `/tmp/mllm-parity-cuda-final-gpu6.log`
- `/tmp/full-digest-pi0-gpu6.log`
- `/tmp/final-mllm-vulkan-gpu6.log`
- `/tmp/final-audit-pi0-vulkan-gpu6.log`
- `/tmp/final-audit-llm-vulkan-gpu6.log`
- `/tmp/final-audit-mllm-vulkan-gpu6.log`
- `/tmp/final-opencl-hard-gate.log`
- `/tmp/final-sycl-gate3.log`
- `/tmp/decode-budget-llm-gpu6.log`
- `/tmp/decode-budget-mllm-gpu6.log`
- `/tmp/llm-cuda-perf-gpu6.log`
- `/tmp/llm-cuda-memory-gpu6.csv`
- `/tmp/llm-sycl-perf-gpu6.log`
- `/tmp/llm-sycl-memory-gpu6.csv`
- `/tmp/pi0-engine-sycl-allweights-gpu6.log`
- `/tmp/pi0-parity-sycl-gpu6.log`
- `/tmp/mllm-cpu-perf-final.log`
- `/tmp/mllm-cuda-perf-gpu6.log`
- `/tmp/mllm-sycl-perf-gpu6.log`
- `/tmp/final-sycl-upscale-backend-ops-gpu6.log`
- `/tmp/llm-opencl-nvidia-gpu6.log`
- `/tmp/final-backend-dl-llm-pure-cpu.log`
- `/tmp/final-backend-dl-llm-cuda-gpu6.log`
- `/tmp/final-backend-dl-python-llm-cuda-gpu6.log`
- `/tmp/final-prod-direct-python-llm-cuda-gpu6.log`
- `/tmp/final-cuda-provider-ctest-gpu6.log`
- `/tmp/final-sycl-provider-ctest-gpu6.log`
- `/tmp/final-python-{pi0,llm,mllm}-cuda-gpu6.log`
- `/tmp/final-python-pi0-sycl-gpu6.log`
- `/tmp/final-review-fix-cuda-provider-ctest-gpu6.log`
- `/tmp/review-fix-sycl-provider-ctest-gpu6.log`
- `/tmp/review-fix-python-{llm,mllm}-cuda-gpu6.log`
- `/tmp/review-fix-backend-dl-llm-pure-cpu.log`
- `/tmp/review-fix-backend-dl-llm-cuda-gpu6.log`
- `/tmp/review-fix-backend-dl-python-llm-cuda-gpu6.log`
- `/tmp/review-fix-prod-direct-python-llm-cuda-gpu6.log`
- `/tmp/review-fix-nonmodel-ctest.log`
- `/tmp/pi0-cli-baseline-gpu6.log`
- `/tmp/pi0-cli-parity-gpu6.log`
- `/tmp/pi0-cli-parity-gate-gpu6.log`
- `/tmp/pi0-flashrt-cli-parity-gpu6.log`

## Reproduction

Check for an idle device first; prefer GPU 6 when several devices are idle.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

export CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH="$PWD/cpp/build-jetson-pi-cuda/bin:$PWD/cpp/build-jetson-pi-cuda/runtime:$LD_LIBRARY_PATH"

FLASHRT_LLM_MODEL=/data/pretrained_models/qwen3-0.6b-q4_k_m.gguf \
FLASHRT_LLM_BACKEND=cuda \
cpp/build-jetson-pi-cuda/test_llama_cpp_jetson_pi_llm

FLASHRT_PI0_MODEL=/data/pretrained_models/pi0_model/pi0_base/Pi0_Base-2.8B-F16.gguf \
FLASHRT_PI0_MMPROJ=/data/pretrained_models/pi0_model/pi0_base/vit/mmproj-model-f16.gguf \
FLASHRT_PI0_FIXTURE_DIR=/tmp/pi0_fixture \
FLASHRT_PI0_ACTION_STEPS=10 FLASHRT_PI0_ACTION_DIM=32 \
FLASHRT_PI0_BACKEND=cuda \
cpp/build-jetson-pi-cuda/test_llama_cpp_jetson_pi_engine

FLASHRT_MLLM_MODEL=/data/pretrained_models/Qwen3-VL-2B-Instruct-GGUF/Qwen3-VL-2B-Instruct-Q4_K_M.gguf \
FLASHRT_MLLM_MMPROJ=/data/pretrained_models/Qwen3-VL-2B-Instruct-GGUF/mmproj-F16.gguf \
FLASHRT_MLLM_BACKEND=cuda \
cpp/build-jetson-pi-cuda/test_llama_cpp_jetson_pi_mllm
```

### Native Pi0 CLI parity

The Pi0 GGUF has no embedded chat template, so the CLI requires an explicit
template rather than choosing one implicitly. For the numerical gate, the
test-only Jinja expression rotates the two 11-byte media markers inserted by
interactive mode from the front to the end. The resulting token input is
exactly the narrow API's `prompt + marker + marker` convention.

From the migration workspace root:

```bash
state="$(od -An -t f4 -v /tmp/pi0_fixture/state.bin | \
  tr -s ' ' '\n' | sed '/^$/d' | paste -sd, -)"
prompt="$(tr -d '\r\n' < /tmp/pi0_fixture/prompt.txt)"

printf '/image %s\n/image %s\n/state %s\n%s\n/exit\n' \
  /tmp/pi0_fixture/image.png /tmp/pi0_fixture/wrist_image.png \
  "$state" "$prompt" | \
CUDA_VISIBLE_DEVICES=6 Jetson-PI/build-backend-dl-cuda/bin/llama-mtmd-cli \
  -m /data/pretrained_models/pi0_model/pi0_base/Pi0_Base-2.8B-F16.gguf \
  --mmproj /data/pretrained_models/pi0_model/pi0_base/vit/mmproj-model-f16.gguf \
  -ngl 999 --jinja \
  --chat-template "{{ messages[0]['content'][22:] }}{{ messages[0]['content'][:22] }}" \
  -v > /tmp/pi0-cli-parity-gpu6.log 2>&1

CUDA_VISIBLE_DEVICES=6 \
FLASHRT_PI0_MODEL=/data/pretrained_models/pi0_model/pi0_base/Pi0_Base-2.8B-F16.gguf \
FLASHRT_PI0_MMPROJ=/data/pretrained_models/pi0_model/pi0_base/vit/mmproj-model-f16.gguf \
FLASHRT_PI0_FIXTURE_DIR=/tmp/pi0_fixture \
FLASHRT_PI0_ACTION_STEPS=10 FLASHRT_PI0_ACTION_DIM=32 \
FLASHRT_PI0_BACKEND=cuda \
FLASHRT_PI0_CLI_ACTION_LOG=/tmp/pi0-cli-parity-gpu6.log \
FlashRT/cpp/build-jetson-pi-cuda/test_llama_cpp_jetson_pi_parity
```

The final GPU6 gate reports 320 CLI elements and
`CLI max abs diff = 0`, then verifies byte identity with `memcmp`.

The Vulkan, OpenCL, and SYCL build commands and runtime dependency notes are
recorded in `docs/jetson_pi_usage.md`.
