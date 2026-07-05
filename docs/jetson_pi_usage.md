# Jetson-PI Pi0 provider (Python)

`flash_rt.load_model(..., framework="jetson_pi")` drives the Jetson-PI
llama.cpp/GGML Pi0 provider through the FlashRT `frt_model_runtime_v2` C ABI
via ctypes. No torch/jax, no GPU arch detection — the Pi0 whole-graph infer
runs in-process on the Jetson-PI `jetson_pi_pi0` policy library.

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
    action_steps=10,          # pi0_base; pi0_libero_base is 50
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

## Limitations (Phase 2)

- **Pi0 only.** Generic GGUF LLM is Phase 3; multimodal LLM is Phase 4.
- **Raw action chunk.** `predict` returns the model's `action_steps ×
  action_dim` output without unnormalization or LIBERO 7-D slicing. The caller
  is responsible for post-processing (use `meta/stats.json` to unnormalize).
- **CPU backend verified.** `backend="cuda"` is wired through but not yet
  tested end-to-end on this machine.
- **No calibration.** The frontend has no `calibrate`/`calibrated`; the
  Jetson-PI provider does not need FlashRT-style FP8 calibration.
- **`state` is a separate port**, not encoded into the prompt (unlike Pi0.5).
  `VLAModel.predict` detects that `set_prompt` does not accept `state` and
  routes it through `observation["state"]` automatically.
