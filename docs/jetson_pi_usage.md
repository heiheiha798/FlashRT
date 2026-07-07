# Jetson-PI providers (Python)

`flash_rt.load_model(..., framework="jetson_pi")` drives Jetson-PI
llama.cpp/GGML providers through the FlashRT `frt_model_runtime_v2` C ABI
via ctypes. No torch/jax, no GPU arch detection.

Three providers, selected by `config=`:
- **`config="pi0"`** (default for VLA) — Pi0 whole-graph infer via
  `jetson_pi_pi0`. Returns a `VLAModel` (`.predict(images, prompt, state)`).
- **`config="llm"`** — generic GGUF text completion via `jetson_pi_llm`.
  Returns an `LlmJetsonPiFrontend` directly (`.generate(prompt)`); not a VLA.
- **`config="mllm"`** — multimodal LLM (vision+text) via `jetson_pi_mllm`.
  Returns an `MllmJetsonPiFrontend` directly (`.generate(images, prompt)`);
  not a VLA.

## Build

```bash
cmake -S FlashRT/cpp -B FlashRT/cpp/build-jetson-pi \
  -DCMAKE_C_COMPILER=.../x86_64-conda-linux-gnu-cc \
  -DCMAKE_CXX_COMPILER=.../x86_64-conda-linux-gnu-c++ \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DFLASHRT_CPP_WITH_JETSON_PI=ON \
  -DJETSON_PI_ROOT=/path/to/Jetson-PI
cmake --build FlashRT/cpp/build-jetson-pi -j32 --target flashrt_cpp_llama_cpp_provider_c
```

Produces `FlashRT/cpp/build-jetson-pi/libflashrt_cpp_llama_cpp_provider_c.so`.

## Usage

```python
import flash_rt

model = flash_rt.load_model(
    "/path/to/Pi0_Base-2.8B-F16.gguf",
    framework="jetson_pi",
    mmproj_path="/path/to/mmproj-model-f16.gguf",
    backend="cpu",            # or "cuda"
    num_views=2,
    action_steps=10,          # pi0_base metadata; see CLAUDE.md weights note
                             # (pi0_libero_base fails this fork's check_tensor_dims)
    action_dim=32,
    lib_path=None,            # auto-discover, or set FLASHRT_PI0_LIB
)

actions = model.predict(
    [image_rgb_224x224, wrist_rgb_224x224],   # list of HxWx3 uint8 numpy
    prompt="put the mug on the plate",
    state=robot_state_floats,                  # 1-D float32, ≤ action_dim
)
# actions: np.ndarray shape (action_steps, action_dim), float32, raw (no unnorm)
```

`LD_LIBRARY_PATH` must include the conda lib dir (for `libgomp.so`, needed by
`libggml-cpu.so`) and the build dir (for `libjetson_pi_pi0.so` /
`libmtmd.so` / `libllama.so` / `libggml*.so`):

```bash
export LD_LIBRARY_PATH=.../miniconda3/lib:FlashRT/cpp/build-jetson-pi
```

## Run the smoke test

```bash
python FlashRT/cpp/tests/fixtures/prepare_pi0_fixture.py \
  --npz <libero obs.npz> --prompt <prompt.txt> --out-dir /tmp/pi0_fixture --action-dim 32

FLASHRT_PI0_MODEL=.../Pi0_Base-2.8B-F16.gguf \
FLASHRT_PI0_MMPROJ=.../mmproj-model-f16.gguf \
FLASHRT_PI0_FIXTURE_DIR=/tmp/pi0_fixture \
FLASHRT_PI0_LIB=FlashRT/cpp/build-jetson-pi/libflashrt_cpp_llama_cpp_provider_c.so \
FLASHRT_PI0_ACTION_STEPS=10 FLASHRT_PI0_ACTION_DIM=32 \
python -m flash_rt.tests.test_jetson_pi_pi0_python
```

## Limitations
- **Pi0 raw action chunk.** `predict` returns the model's `action_steps ×
  action_dim` output without unnormalization or LIBERO 7-D slicing. The caller
  is responsible for post-processing (use `meta/stats.json` to unnormalize).
- **LLM raw prompt.** `generate(prompt)` takes a raw prompt; the caller must
  apply the chat template (e.g. `llama_chat_apply_template`) before calling.
  No streaming; one blob out. Each call clears KV (independent completion).
- **CPU, CUDA, and Vulkan backends verified.** `backend="cuda"` is verified
  end-to-end on an RTX 4090 (sm_89) for all three providers — Pi0 (`offloaded
  37/37 layers`, pi0_base), LLM (`29/29`, qwen3-0.6b-q4_k_m), and MLLM
  (`37/37`, Qwen2.5-VL-3B-Instruct-q4_0; the VIT/mmproj encoder offloads too).
  For Pi0 and LLM the generated output is coherent; for MLLM the GPU execution
  path (load → VIT encode → forward → sample) is exercised, but the smoke test
  feeds a raw prompt so the output is template-token noise rather than a
  caption — output coherence requires a caller-applied chat template (see the
  MLLM note below). `backend="vulkan"` is also verified on the RTX 4090 for
  Pi0 and LLM (smoke + numerical parity vs the direct `jetson_pi_*` call, both
  passing with `max_diff = 0` / exact text match); the VIT/mmproj encoder
  offloads to Vulkan0 too (`clip_ctx: CLIP using Vulkan0 backend`). Per §6 of
  the migration plan, compiling a backend does not guarantee every model op is
  supported on it — each model×backend combo is verified separately. The CUDA
  build needs its own build dir with `-DGGML_CUDA=ON` (it defaults OFF), plus
  the same Jetson-PI flags:

  ```bash
  cmake -S FlashRT/cpp -B FlashRT/cpp/build-jetson-pi-cuda \
    -DCMAKE_C_COMPILER=.../x86_64-conda-linux-gnu-cc \
    -DCMAKE_CXX_COMPILER=.../x86_64-conda-linux-gnu-c++ \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    -DFLASHRT_CPP_WITH_JETSON_PI=ON \
    -DJETSON_PI_ROOT=/path/to/Jetson-PI \
    -DGGML_CUDA=ON -DGGML_CUDA_FA=ON
  cmake --build FlashRT/cpp/build-jetson-pi-cuda -j32 \
    --target flashrt_cpp_llama_cpp_provider_c
  ```

  Then point `FLASHRT_PI0_LIB` at `build-jetson-pi-cuda/libflashrt_cpp_llama_cpp_provider_c.so`,
  set `FLASHRT_PI0_BACKEND=cuda`, and ensure `LD_LIBRARY_PATH` includes the
  CUDA build's `bin/` (for `libggml-cuda.so`) and `runtime/`. The smoke test
  then offloads all model layers to the GPU (37/37 for pi0_base). The same
  `_c.so` and env-override recipe apply to the LLM (`FLASHRT_LLM_*`) and MLLM
  (`FLASHRT_MLLM_*`) smoke tests — see their sections below.

  Note: the Jetson-PI engine accepts `backend` `"cpu"` / `"cuda"` / `"vulkan"`
  by exact string match; any other value (including typos) is **rejected** by
  `open` (returns `INVALID` + a null handle), NOT a silent CPU fallback — this
  honors the project's no-fallback contract. `"cpu"` sets `n_gpu_layers=0`;
  `"cuda"`/`"vulkan"` set `n_gpu_layers=9999`. The actual device (CUDA0 /
  Vulkan0) is chosen by GGML's scheduler from the backends registered by
  `ggml_backend_load_all` (i.e. what was compiled in), **not** pinned by the
  string — so on a CUDA+Vulkan mixed build, `backend="vulkan"` would split
  layers across CUDA0+Vulkan0, and on a build lacking the requested backend GGML
  leaves everything on CPU. The per-build recipes below use a single-backend
  build (`-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON -DGGML_CUDA=OFF`) to avoid that
  ambiguity. Treat the `load_tensors: layer N assigned to device <Dev>` /
  `offloaded N/N layers to <Dev>` log line as the real signal that the intended
  backend was exercised.

  **Vulkan backend** (verified on RTX 4090 for Pi0 + LLM). Build in a separate
  dir with `-DGGML_VULKAN=ON -DGGML_CUDA=OFF` (Vulkan is SPIR-V; no CUDA arch
  needed) and conda compilers; the `vulkan-shaders-gen` build-time tool needs a
  C++17 host compiler, so export `CC`/`CXX` to the conda compilers for the
  build step:

  ```bash
  cmake -S FlashRT/cpp -B FlashRT/cpp/build-jetson-pi-vulkan \
    -DCMAKE_C_COMPILER=.../x86_64-conda-linux-gnu-cc \
    -DCMAKE_CXX_COMPILER=.../x86_64-conda-linux-gnu-c++ \
    -DCMAKE_PREFIX_PATH=.../envs/migrate \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    -DFLASHRT_CPP_WITH_JETSON_PI=ON \
    -DJETSON_PI_ROOT=/path/to/Jetson-PI \
    -DGGML_VULKAN=ON -DGGML_CUDA=OFF
  CC=.../x86_64-conda-linux-gnu-cc CXX=.../x86_64-conda-linux-gnu-c++ \
    cmake --build FlashRT/cpp/build-jetson-pi-vulkan -j32 \
    --target flashrt_cpp_llama_cpp_provider_c
  ```

  `CMAKE_PREFIX_PATH` must point at a prefix providing the Vulkan loader
  (`libvulkan.so`), `glslc` (conda-forge `vulkan-loader` + `shaderc`), and the
  Vulkan **headers**. Header caveat: conda-forge `vulkan-headers` (1.3.231)
  lacks the `vk_video/` av1 codec headers and `vulkan_hpp_macros.hpp` that
  `vulkan_core.h` unconditionally `#include`s, so a build against just the conda
  headers fails with `fatal error: vk_video/vulkan_video_codec_av1std.h`. Use
  the current Khronos `Vulkan-Headers` (≥1.3.295, which bundles `vk_video/` +
  the split hpp files) — `git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git`
  and point `CMAKE_PREFIX_PATH` at its `include` parent (or stage its `vulkan/`
  + `vk_video/` ahead of the conda env's). Do NOT rely on overwriting the conda
  env's `include/vulkan` in place — a later `conda install/update vulkan-headers`
  clobbers it; a separate prefix is robust. The Vulkan device is selected by
  `GGML_VK_VISIBLE_DEVICES=<idx>` (emulates `CUDA_VISIBLE_DEVICES`). Then set
  `FLASHRT_PI0_BACKEND=vulkan` / `FLASHRT_LLM_BACKEND=vulkan`, point
  `FLASHRT_PI0_LIB`/`FLASHRT_LLM_LIB` at the vulkan build's `_c.so`, and ensure
  `LD_LIBRARY_PATH` includes the vulkan build's `bin/` (for `libggml-vulkan.so`).
  Verified: Pi0 `pi0_base` offloads 36 layers + VIT to Vulkan0, actions
  (10,32) sane, parity `max_diff = 0` vs the direct `jetson_pi_pi0` call;
  LLM `qwen3-0.6b-q4_k_m` offloads 28 layers to Vulkan0, greedy text
  byte-identical to the direct `jetson_pi_llm` call. MLLM-on-Vulkan is not
  verified this round (§6: each model×backend combo separately).
- **No calibration.** The frontends have no `calibrate`/`calibrated`; the
  Jetson-PI providers do not need FlashRT-style FP8 calibration.
- **`state` is a separate port** for Pi0, not encoded into the prompt (unlike
  Pi0.5). `VLAModel.predict` detects that `set_prompt` does not accept `state`
  and routes it through `observation["state"]` automatically.

## Generic GGUF LLM (Phase 3)

```python
import flash_rt

fe = flash_rt.load_model(
    "/path/to/qwen3-0.6b-q4_k_m.gguf",
    framework="jetson_pi",
    config="llm",
    backend="cpu",
    n_ctx=2048, n_threads=0,
    temp=0.8, top_k=40, top_p=0.9, seed=1, max_tokens=512,
    lib_path=None,            # auto-discover, or set FLASHRT_LLM_LIB
)

text = fe.generate("What is 2 plus 2? The answer is")
# text: str, the generated completion (no chat template applied by the engine)
```

The returned object is an `LlmJetsonPiFrontend` (not a `VLAModel` — LLMs are
not VLA). `fe.infer({"prompt": ...})` returns `{"text": ...}` for callers that
want a dict-shaped interface.

### Run the LLM smoke test

```bash
FLASHRT_LLM_MODEL=.../qwen3-0.6b-q4_k_m.gguf \
FLASHRT_LLM_LIB=FlashRT/cpp/build-jetson-pi/libflashrt_cpp_llama_cpp_provider_c.so \
LD_LIBRARY_PATH=.../miniconda3/lib:FlashRT/cpp/build-jetson-pi \
python -m flash_rt.tests.test_jetson_pi_llm_python
```

`FLASHRT_LLM_BACKEND=cuda` switches the test to the GPU forward pass; point
`FLASHRT_LLM_LIB` at the `build-jetson-pi-cuda/libflashrt_cpp_llama_cpp_provider_c.so`
and add the cuda build's `bin/` to `LD_LIBRARY_PATH` (same recipe as the Pi0
CUDA section above). Verified on an RTX 4090 (sm_89): `offloaded 29/29 layers
to GPU` for qwen3-0.6b-q4_k_m.

## Multimodal LLM (Phase 4)

```python
import flash_rt

fe = flash_rt.load_model(
    "/path/to/Qwen2.5-VL-3B-Instruct-q4_0.gguf",
    framework="jetson_pi",
    config="mllm",
    mmproj_path="/path/to/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf",
    backend="cpu",
    n_ctx=2048, n_threads=0,
    temp=0.0, top_k=0, top_p=0.0, seed=1, max_tokens=64,
    lib_path=None,            # auto-discover, or set FLASHRT_MLLM_LIB
)

text = fe.generate(
    [image_rgb_224x224],      # list of HxWx3 uint8 numpy (may be empty for text-only)
    "Describe this image in one sentence.",
)
# text: str, the generated completion
```

The returned object is an `MllmJetsonPiFrontend` (not a `VLAModel` — MLLMs
output text, not actions). `fe.infer({"images": [...], "prompt": ...})` returns
`{"text": ...}` for callers that want a dict-shaped interface.

The caller is responsible for applying the chat template; the engine only does
raw prompt + media markers → text. Each `generate` call clears KV (independent
completion, no multi-turn).

### Run the MLLM smoke test

```bash
FLASHRT_MLLM_MODEL=.../Qwen2.5-VL-3B-Instruct-q4_0.gguf \
FLASHRT_MLLM_MMPROJ=.../Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
FLASHRT_MLLM_LIB=FlashRT/cpp/build-jetson-pi/libflashrt_cpp_llama_cpp_provider_c.so \
LD_LIBRARY_PATH=.../miniconda3/lib:FlashRT/cpp/build-jetson-pi \
python -m flash_rt.tests.test_jetson_pi_mllm_python
```

`FLASHRT_MLLM_BACKEND=cuda` switches the test to the GPU forward pass (LLM
layers + VIT/mmproj encoder both offloaded); point `FLASHRT_MLLM_LIB` at the
`build-jetson-pi-cuda/libflashrt_cpp_llama_cpp_provider_c.so` (same CUDA recipe
as the Pi0 section above). Verified on an RTX 4090 (sm_89): `offloaded 37/37
layers to GPU` for Qwen2.5-VL-3B-Instruct-q4_0.

Note: the engine injects one media marker per supplied image *after* the prompt
text (see `jetson_pi_mllm.cpp`); the smoke test feeds a raw prompt, so the
generated text is a connectivity check (non-empty, printable), not a
task-accurate caption. Callers wanting a well-formed answer must apply the
model's chat template (with the image placeholder positioned by the template)
themselves.

