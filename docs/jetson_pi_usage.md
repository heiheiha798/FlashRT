# Jetson-PI Provider

The Jetson-PI integration is an optional llama.cpp/GGML model provider. It
publishes the standard `frt_model_runtime_v1` contract and the selected generic
stage-plan extension. FlashRT runtime code contains no Jetson-PI or model-family
branches.

## Build

The provider is disabled by default. Install Jetson-PI first, then install
FlashRT into the same prefix so the provider and its dependency closure are
co-located:

```bash
cmake -S cpp -B build/provider \
  -DFLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER=ON \
  -DFLASHRT_CPP_WITH_JETSON_PI=ON \
  -DJetsonPI_ROOT=<jetson-pi-install-prefix>
cmake --build build/provider --target flashrt_cpp_llama_cpp_provider_c
cmake --install build/provider --prefix <install-prefix>
```

`<jetson-pi-install-prefix>` and `<install-prefix>` should identify the same
deployment prefix. FlashRT does not copy or take ownership of third-party
libraries.

For development against a Jetson-PI source tree, use
`-DJETSON_PI_ROOT=<jetson-pi-source-root>` instead. Set exactly one of
`JETSON_PI_ROOT` and `JetsonPI_ROOT`.

`FLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER=ON` without
`FLASHRT_CPP_WITH_JETSON_PI=ON` builds the provider contract and fake-engine
tests without linking Jetson-PI. This mode is suitable for CPU-only contract
CI.

The installed provider DSO exports one entry point:

```c
int frt_model_runtime_open_v1(
    const char* config_json, frt_model_runtime_v1** out);
```

All provider and backend implementation symbols remain hidden.

## Runtime Contract

The returned object always uses model-runtime v1:

- ports describe model inputs and outputs;
- `step()` performs one whole-model request;
- `FRT_EXT_GENERIC_STAGE_PLAN_V1` publishes one selected stage plan;
- Jetson-PI state, KV caches, samplers, and intermediate tensors remain
  provider-private;
- outputs use ordinary STAGED `get_output`; no provider-specific buffer ABI is
  required.

Engines with staged capability publish `context -> action` for VLA models or
`reset -> prefill -> decode` for autoregressive models. Engines without that
capability publish a single `infer` stage. The plan uses OPAQUE executors, so a
generic host never sees llama.cpp or GGML types.

## Python

### VLA

```python
import flash_rt

model = flash_rt.load_model(
    "policy.gguf",
    framework="jetson_pi",
    config="pi0",
    mmproj_path="vision.gguf",
    backend="cuda",
    num_views=2,
    action_steps=10,
    action_dim=32,
)

actions = model.predict(images, prompt=prompt, state=state)
```

`images` is an ordered list of contiguous RGB `uint8` HWC arrays. `state` is a
contiguous `float32` vector with `action_dim` values. The result is a copied
`float32[action_steps, action_dim]` array.

The staged face is available through the frontend:

```python
model._pipe.context({"images": images, "state": state, "prompt": prompt})
actions = model._pipe.action()
```

### Text LLM

```python
llm = flash_rt.load_model(
    "model.gguf",
    framework="jetson_pi",
    config="llm",
    backend="cuda",
    max_tokens=128,
)

text = llm.generate("Write a short summary.")
```

For host-driven decoding, call `reset()`, `prefill()`, and `decode()`. The host
may stop by not issuing another decode call.

### Multimodal LLM

```python
mllm = flash_rt.load_model(
    "model.gguf",
    framework="jetson_pi",
    config="mllm",
    mmproj_path="vision.gguf",
    backend="cuda",
    max_tokens=128,
)

text = mllm.generate(images, "Describe the scene.")
```

## C++ Host

Load the provider DSO, resolve `FRT_MODEL_RUNTIME_OPEN_V1_SYMBOL`, and probe the
returned object using `FRT_MODEL_RUNTIME_V1_BASE_SIZE`. Probe
`query_extension` only when `struct_size` reaches
`FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE`.

When `FRT_EXT_GENERIC_STAGE_PLAN_V1` is present:

1. Validate the extension version and size.
2. Validate the dependency DAG.
3. Stage input ports through `set_input`.
4. Execute the selected plan in dependency order.
5. Read staged outputs through `get_output`.
6. Release the model runtime once.

Nexus provides the generic host-side adopter for this flow. Provider code owns
model execution; Nexus owns lifecycle and scheduling policy.

## Failure Rules

- Unknown or duplicate JSON fields are rejected.
- Missing checkpoints fail before engine creation.
- Unsupported backends fail instead of falling back silently.
- Failed input replacement invalidates the old ready state.
- Stage ordering and decode budgets are enforced by the provider.
- Provider identity includes checkpoint digests, backend, port schema, and the
  selected stage plan.
