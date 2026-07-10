# Jetson-PI Provider Validation Matrix

This document is the reviewable record for `jetsonpi迁移.txt` §14. It
separates correctness gates from diagnostic performance measurements. Timings
are single-run wall-clock diagnostics from the named logs, not statistically
stable benchmarks and not cross-backend speed claims.

## Validated Revisions

- FlashRT base revision for this final validation change:
  `046cbce4783d7121e428d530b2c61309a80126de` plus the reviewed working tree.
- Jetson-PI revision:
  `a4119a1731017635f83f29a03eef676c7d76c3b9`.
- CUDA/Vulkan device used for the final accelerator runs: physical GPU 6,
  NVIDIA GeForce RTX 4090. CUDA used `CUDA_VISIBLE_DEVICES=6`; Vulkan used
  `GGML_VK_VISIBLE_DEVICES=6`.

## Correctness Matrix

| Model face | Model | Backend | Result | Evidence |
|---|---|---|---|---|
| Pi0 | Pi0 Base F16 + VIT mmproj | CPU | PASS | FlashRT versus direct `jetson_pi_pi0` action parity, `max_abs_diff=0`; shape `(10,32)`; finite actions. |
| Pi0 | Pi0 Base F16 + VIT mmproj | CUDA | PASS | 37/37 model layers and VIT on CUDA; whole infer repeatability; no context leak; FlashRT/direct parity `max_abs_diff=0`; whole versus context/action bit-identical. |
| Pi0 | Pi0 Base F16 + VIT mmproj | Vulkan | PASS | 36 model layers plus VIT on Vulkan; FlashRT/direct parity `max_abs_diff=0`. |
| Text LLM | Qwen3 0.6B Q4_K_M | CPU | PASS | FlashRT/direct greedy text, first token, and complete prefill logits parity; logits `max_abs_diff=0`. |
| Text LLM | Qwen3 0.6B Q4_K_M | CUDA | PASS | 29/29 layers offloaded; FlashRT/direct text, first token, and complete prefill logits parity with `max_abs_diff=0`; two distinct interleaved sessions reproduce standalone token/EOG sequences. |
| Text LLM | Qwen3 0.6B Q4_K_M | Vulkan | PASS | 29/29 layers offloaded; FlashRT/direct greedy text exact parity. |
| Text LLM | TinyLlama 1.1B | CPU/CUDA | PASS | 23/23 CPU and 23/23 CUDA model matrix runs complete. |
| Text LLM | Gemma 2 2B | CPU/CUDA | PASS | 27/27 CPU and 27/27 CUDA model matrix runs complete. |
| MLLM | Qwen3-VL 2B Q4_K_M + F16 mmproj | CPU | PASS | FlashRT and direct `jetson_pi_mllm` produce exact image+text output for the same generated RGB fixture and greedy sampling. |
| MLLM | Qwen3-VL 2B Q4_K_M + F16 mmproj | CUDA | PASS | 29/29 language layers plus CLIP/VIT on CUDA; FlashRT/direct image+text output is exact; one-shot equals staged prefill/decode; finite logits. |
| LLM provider | Qwen3 0.6B | OpenCL | ENVIRONMENT GATE | Build succeeds, but this host exposes no OpenCL platform/device. Explicit `backend="opencl"` hard-fails before model execution. |
| LLM provider | Qwen3 0.6B | SYCL | COMPILE/INSTALL PASS, DEVICE GATE | DPC++ NVIDIA-targeted provider and installed-prefix consumer build successfully. The installed runtime has no NVIDIA CUDA UR adapter, so explicit `backend="sycl"` hard-fails because no accelerator device is registered. |

## Known Driver-Lifecycle Result

- Qwen3-VL 2B inference reaches the success marker with 29/29 language layers
  plus CLIP/VIT on Vulkan. The process then segfaults during NVIDIA 550.54.14
  driver teardown. This is not counted as a clean PASS row, and the provider
  does not install a fallback or speculative teardown workaround.

## §14 Invariants

- Consecutive Pi0 requests invalidate stale actions and reproduce the same
  fixed-seed output for identical inputs; no previous context leaks.
- Pi0 actions are checked for shape, nonzero values, and NaN/Inf absence.
- Pi0 host SWAP is consumed zero-copy through NumPy's versioned, read-only
  DLPack path; pointer identity and the live-consumer stage guard are tested.
  PyTorch 2.6 legacy DLPack export is rejected because it cannot preserve the
  read-only provider mapping, and no implicit copy fallback is used.
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
| Qwen3 0.6B CUDA | Prompt batches observed at about 4-37 ms depending on cold/warm state and prompt length. Repeatable decode is validated per token; use the test command below for current per-run values. |
| Qwen3 0.6B CPU | Full CPU token/logit parity run passed; the direct 12-token prompt batch was about 24.44 s in the final run. |
| TinyLlama 1.1B CPU/CUDA | 12-token batch diagnostics about 2.66-2.79 s on CPU and 4.1-30.6 ms on CUDA. |
| Gemma 2 2B CPU/CUDA | 12-13-token batch diagnostics about 3.92-7.79 s on CPU and 6.5-35.2 ms on CUDA. |
| Qwen3-VL 2B CUDA | 58-59-token multimodal prompt batches about 21-156 ms depending on cold/warm state. |
| Qwen3-VL 2B CPU | Two exact-parity 58-token image+text prompt batches were about 19.04-19.07 s. |
| Qwen3-VL 2B Vulkan | 58-59-token multimodal prompt batches about 39-98 ms before the known process-exit driver teardown failure. |

Memory is checked operationally by successful model offload and completion on
a 24 GiB RTX 4090. The current narrow API does not expose allocator peak bytes,
so this document does not invent a peak-memory number.

Final-run logs used by this matrix include:

- `/tmp/llm-token-logit-parity-cpu-final.log`
- `/tmp/llm-token-logit-parity-cuda-final-gpu6.log`
- `/tmp/mllm-parity-cpu-final.log`
- `/tmp/mllm-parity-cuda-final-gpu6.log`
- `/tmp/full-digest-pi0-gpu6.log`
- `/tmp/final-mllm-vulkan-gpu6.log`
- `/tmp/final-opencl-hard-gate.log`
- `/tmp/final-sycl-gate3.log`
- `/tmp/decode-budget-llm-gpu6.log`
- `/tmp/decode-budget-mllm-gpu6.log`

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

The Vulkan, OpenCL, and SYCL build commands and runtime dependency notes are
recorded in `docs/jetson_pi_usage.md`.
