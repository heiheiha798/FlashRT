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

Two integration routes (§5.2): a **dev** `add_subdirectory` path and a
**production** `find_package` path. Pick one.

### Dev: `add_subdirectory` (builds Jetson-PI inline from its source tree)

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

### Production: `find_package(JetsonPI)` (link a pre-installed Jetson-PI)

Build and install Jetson-PI as a standalone tree (NOT via FlashRT's
`add_subdirectory`). With `LLAMA_BUILD_COMMON=ON` + `JETSON_PI_BUILD_MTMD=ON`
the narrow `jetson_pi_*` libs and `mtmd`/`llama`/`ggml` deps build; install lays
down `lib64/libjetson_pi_{pi0,llm,mllm}.so`, `lib64/lib{llama,mtmd,ggml*}.so`, and
`include/jetson_pi_{pi0,llm,mllm}.h`:

```bash
cmake -S Jetson-PI -B Jetson-PI/build-prod \
  -DCMAKE_C_COMPILER=.../x86_64-conda-linux-gnu-cc \
  -DCMAKE_CXX_COMPILER=.../x86_64-conda-linux-gnu-c++ \
  -DCMAKE_CUDA_COMPILER=.../nvcc -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DLLAMA_BUILD_COMMON=ON -DJETSON_PI_BUILD_MTMD=ON \
  -DLLAMA_CURL=OFF -DLLAMA_HTTPLIB=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_TOOLS_INSTALL=OFF \
  -DCMAKE_INSTALL_PREFIX=/path/to/jetson-pi-install-prefix
cmake --build Jetson-PI/build-prod -j16 \
  --target jetson_pi_pi0 jetson_pi_llm jetson_pi_mllm
cmake --install Jetson-PI/build-prod
```

> `LLAMA_TOOLS_INSTALL=OFF` is load-bearing: with `JETSON_PI_BUILD_MTMD=ON`,
> `tools/mtmd/CMakeLists.txt` unconditionally `add_executable(llama-mtmd-cli)`
> but only builds it when the full tools tree is built; the CLI's
> `install(TARGETS)` rule still runs unless `LLAMA_TOOLS_INSTALL=OFF`, aborting
> `cmake --install` with `cannot find llama-mtmd-cli` — and that abort happens
> *before* `libllama.so` is laid down, so `find_package(JetsonPI)` then fails on
> the missing `llama` lib.

Then configure FlashRT against the installed prefix (no `JETSON_PI_ROOT`):

```bash
cmake -S FlashRT/cpp -B FlashRT/cpp/build-jetson-pi-prod \
  -DCMAKE_C_COMPILER=.../x86_64-conda-linux-gnu-cc \
  -DCMAKE_CXX_COMPILER=.../x86_64-conda-linux-gnu-c++ \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DFLASHRT_CPP_WITH_JETSON_PI=ON \
  -DJetsonPI_ROOT=/path/to/jetson-pi-install-prefix
cmake --build FlashRT/cpp/build-jetson-pi-prod -j32 --target flashrt_cpp_llama_cpp_provider_c
```

`FlashRT/cmake/FindJetsonPI.cmake` resolves the installed libs/headers into
imported targets `JetsonPI::jetson_pi_pi0` / `::jetson_pi_llm` / `::jetson_pi_mllm`
(+ `::mtmd` / `::llama` / `::ggml` / per-backend); `ldd` on the built
`libflashrt_cpp_llama_cpp_provider_c.so` must then resolve to the installed
prefix's `libjetson_pi_pi0.so.0` / `libllama.so.0` / `libggml-cuda.so.0`, not to
any in-tree copy. `cmake --install` of FlashRT lays down the SHARED provider
`_c.so` and `libflashrt_runtime.so` to `${CMAKE_INSTALL_LIBDIR}`. The installed
provider has an install RPATH containing `$ORIGIN` plus the exact library
directories resolved under `JetsonPI_ROOT`, so it can be loaded without an
ambient `LD_LIBRARY_PATH`. Set `-DJETSON_PI_ROOT=<source>` for dev OR
`-DJetsonPI_ROOT=<prefix>` for prod — exactly one is required. Configuration
fails explicitly when neither or both are set, preventing an implicit search
from selecting a stale system installation.

When the installed prefix contains `libggml-cuda`, FlashRT's production
configure requires the matching CUDA Toolkit CMake package so the imported
`ggml-cuda` target carries its driver/runtime/cuBLAS link interface. This is a
build-host requirement; the installed provider itself was verified with only
its recorded RPATH and the normal NVIDIA runtime loader environment.

> **The Pi0 parity + engine tests are dev-path only.** `test_llama_cpp_jetson_pi_parity`
> and `test_llama_cpp_jetson_pi_engine` `#include "stb/stb_image.h"`, which lives
> only in Jetson-PI's `vendor/` source tree; the prod path has no `JETSON_PI_ROOT`,
> so `cpp/CMakeLists.txt` does not define those two targets on the prod path.
> The link, LLM, MLLM, and LLM-parity tests do not use stb and remain available
> with prod `BUILD_TESTING=ON`. A prod-path GPU smoke via the Python frontend
> (`FLASHRT_PI0_LIB=<installed _c.so>`) exercises the installed artifact
> end-to-end instead — verified on GPU6/RTX 4090 on 2026-07-10 (`offloaded
> 37/37`, actions `(10,32)`, no NaN/Inf, nonzero; installed `_c.so` resolves
> `libflashrt_runtime.so` through `$ORIGIN` and Jetson-PI/mtmd/llama/GGML through
> its exact install-prefix RPATH, without `LD_LIBRARY_PATH`).

### Pinned Jetson-PI commit (§15.1)

The Jetson-PI fork must be used as a locked version with its own GGML — do not
mix another llama.cpp/GGML version into the same provider (ABI/symbol/layout
conflicts). This integration is developed and GPU-verified against Jetson-PI
commit `1450cec` (branch `expose-mtmd-only-build`, local — not pushed to a
remote). When upgrading the fork, re-run the parity tests
(`test_llama_cpp_jetson_pi_parity` / `test_llama_cpp_jetson_pi_llm_parity`) to
confirm FlashRT's provider path still matches the direct `jetson_pi_*` path
byte-for-byte.

### Symbol visibility (§15.2)

Each narrow C API lib (`libjetson_pi_pi0` / `libjetson_pi_llm` /
`libjetson_pi_mllm`) exports only its `jetson_pi_*` C symbols — no `ggml_*` /
`llama_*` / `mtmd_*` C API symbols are defined in the narrow libs (they are
imported via the PUBLIC link to `mtmd`/`llama`/`ggml`). Verified by
`for lib in libjetson_pi_{pi0,llm,mllm}.so; do nm -D --defined-only "$lib" |
grep -E ' (ggml_|llama_|mtmd_)[A-Za-z]'; done` → empty. (A
`std::vector<mtmd_bitmap*>` stdlib template weak symbol appears in
the pi0/mllm libs; that is a benign COMDAT stdlib instantiation referencing the
`mtmd_bitmap` type, not the mtmd C API.)


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

# Optional fine-grained execution. Context owns prompt/image encoding and
# provider-private encoded KV; action consumes that pending context once.
model._pipe.set_prompt("put the mug on the plate")
model._pipe.context({"images": [image, wrist_image], "state": robot_state})
actions = model._pipe.action()

# Read-only zero-copy host SWAP view over the latest action chunk. NumPy's
# native DLPack protocol lets torch/jax consume the same storage.
actions_view = model._pipe.action_view()
torch_actions = torch.from_dlpack(actions_view)
actions_view.close()  # DLPack consumer keeps its own map alive
del torch_actions     # required before replacing inputs or running a stage
```

The Pi0 runtime exposes callback stages `infer`, `context`, and `action`, with
an explicit `context -> action` dependency. `infer` remains the compatibility
whole-tick face and is implemented by the same narrow context/action API.

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

# Host-driven KV-cache session. prefill accepts exactly one of prompt/tokens.
logits = fe.prefill("What is 2 plus 2? The answer is")
step = fe.decode()  # {"token": int, "is_eog": bool, "text": accumulated str}
logits = fe.get_logits()
fe.reset()
logits = fe.prefill(tokens=[151644, 198])  # caller-owned int32 token IDs
```

The returned object is an `LlmJetsonPiFrontend` (not a `VLAModel` — LLMs are
not VLA). `fe.infer({"prompt": ...})` returns `{"text": ...}` for callers that
want a dict-shaped interface.

The provider runtime exposes callback stages `infer`, `reset`, `prefill`, and
repeatable `decode`, with STAGED `prompt`, optional `tokens`, `next_token`,
`logits`, `is_eog`, and accumulated `text` ports. KV state and sampling remain
provider-private.

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

logits = fe.prefill([image], "Describe this image.")
step = fe.decode()
fe.reset()

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
