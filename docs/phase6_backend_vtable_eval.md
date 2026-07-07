# Phase 6 Evaluation — Backend-neutral exec/backend vtable + memory-domain contract

## 1. Executive summary + recommendation

**Recommendation: `go-minimal-contract` — build a small, append-only memory-domain contract; do NOT build a full backend-neutral exec/backend vtable in this phase.**

Four independent lenses (exec/ coupling depth, provider/zero-copy path needs, mllm reference patterns, ABI/memory-domain constraints) converge on the same conclusion. The decisive findings, all verified against the source:

1. **exec/ already has a backend-neutral vtable SHAPE.** `exec/backend/backend.h` is a flat namespace `frt::be::*` of 13 free functions (malloc/free, stream/event, memcpy_dtod_async, capture_begin/end, graph_launch, graph_exec_destroy). The core ABI in `exec/src/*.cpp` links ONLY these — never a CUDA symbol, zero `virtual`. ALL CUDA Graph API calls are confined to ONE file, `exec/backend/cuda/cuda_backend.cpp` (107 lines). Adding a second exec backend is one new TU + a CMake selection line — `exec/src/` and `exec/include/` change ZERO files. A "full vtable" rewrite of exec/ would be solving a problem that the existing free-function seam already solves at link time.

2. **The provider-owned callback-stage bypass ALREADY EXISTS and is in production use.** `FRT_RT_STAGE_CALLBACK` (`model_runtime.h:109-112`), `frt_runtime_stage_desc_v2` (`:172-179`), the `run_stage` verb (`:232`), and `frt_runtime_builder_create_provider_owned` (`:317`) let a provider-owned runtime host a non-CUDA-Graph stage with NO `frt_ctx`/`frt_graph`/`frt_buffer`. The Jetson-PI Pi0 provider (`cpp/providers/llama_cpp/src/pi0_runtime.cpp`) does NOT include `flashrt/exec.h` (verified by grep: zero includes across `cpp/providers/`) and never touches exec/. CUDA-Graph vs provider-owned coexistence is already achieved by leaving the v1 `stages` array empty when callback stages are present (`model_runtime.cpp:232-235`). CLAUDE.md's "provider-owned stage coexistence with CUDA Graph" constraint is already met — Phase 6 does not need to unify the two paths under one backend abstraction.

3. **GGML is self-contained; FlashRT does not execute GGML ops.** A full exec/backend vtable (alloc/free/stream/event/executable create/replay) for the GGML path would have NO consumer — FlashRT would be mediating execution it does not perform. This directly contradicts CLAUDE.md's "Jetson-PI as model provider, not bare exec/ hardware backend" and "mllm is reference architecture, not transplant material."

4. **The ONLY genuinely missing piece for the stated Phase 6 goal** ("FlashRT 与 GGML 的 zero-copy 或 shared-buffer 路径") is a memory-domain CONTRACT: an opaque FlashRT-owned token + ownership/lifetime/location/copy/sync rules, attachable to a provider-owned port in place of today's forbidden SWAP window. The provider-owned builder today rejects all buffers/SWAP on provider-owned ports (`model_runtime.cpp:203-217`), and the docs explicitly defer this to Phase 6 (`docs/model_runtime_api.md:127-129`: "The builder rejects raw SWAP windows on this path until FlashRT has an explicit memory-domain contract for cross-provider buffers"). This is Phase 5's Gap 2 — the single gap that is unambiguously Phase 6's job.

CLAUDE.md Phase 6 text reads "设计更完整的 backend-neutral exec/backend vtable. **可能**包含... 在 contract 明确**后**再考虑 FlashRT 与 GGML 的 zero-copy 或 shared-buffer 路径." The hedge "可能" (possibly/may) and the "after the contract is clear" ordering are load-bearing: the contract is the prerequisite, and it is the deliverable that unblocks the stated zero-copy goal. The minimal honest deliverable that fulfills Phase 6's intent, unblocks Phase 5's future revisit, preserves Phase 1-4, and leaks no GGML types is the **memory-domain contract** — not a full vtable. The exec/ backend seam is already backend-neutral in shape and CUDA-isolated to one TU; documenting its split (shared memory/stream/event primitives vs CUDA-only graph-capture) is in-scope as clarification, but rewriting it into a registered function-pointer vtable is over-design with no consumer.

## 2. exec/ current-state summary + CUDA Graph coupling (lens 1)

exec/ defines exactly five abstractions, all opaque C-ABI handles backed by plain C++ structs in `exec/src/internal.h`:

- **`frt_ctx`** (`internal.h:57-75`) — stream/event pool + arena owner. Owns all buffers; frees them at `frt_ctx_destroy`.
- **`frt_buffer`** (`internal.h:14-20`) — `{ctx, name, dptr, bytes, owned}`. Pure device-pointer wrapper. `frt_buffer_dptr` is the ONLY way the runtime layer touches device memory.
- **`frt_graph`** (`internal.h:32-43`) — ShapeKey→graph-exec variant table with LRU. The executable is a raw `void* exec` (`internal.h:27-30`), NOT a virtual interface.
- **`frt_plan`** (`internal.h:45-55`) — dumb DAG of `(graph, key, stream_id)` nodes. Graph-only by construction (`plan.cpp:75` unconditionally calls `frt_graph_replay`).
- **`frt_event`** (`internal.h:22-25`) — cross-stream sync, wraps `void* handle`.

There is NO separate allocator abstraction — allocation is two free functions `frt::be::malloc/free`. There is NO stage abstraction in exec/ itself; the exec-level replay primitive is `frt_graph_replay(graph, key, stream_id)` (`graph.cpp:105-113`). "Stages" live one layer up in the model-runtime ABI.

**CUDA Graph coupling is confined to ONE TU.** All `cudaStreamBeginCapture`/`cudaStreamEndCapture`/`cudaGraphInstantiate`/`cudaGraphLaunch`/`cudaGraphExecDestroy` calls are in `exec/backend/cuda/cuda_backend.cpp` (`:78-103`). The graph object is a raw `cudaGraphExec_t` cast to `void*` — NOT wrapped in a FlashRT type. The core ABI (`src/*.cpp`) only ever holds it as `void* exec` and passes it to `frt::be::*`. This is the key design property: the core never sees CUDA types, only the backend TU does.

**Where stage replay assumes CUDA Graph** — exactly two sites, BOTH in the model-runtime layer (NOT in exec/): `cpp/models/pi05/src/model_runtime.cpp:231` (`step()` → `frt_graph_replay` for every v1 stage) and `cpp/models/pi05/src/runtime.cpp:203` (`default_replay` → `frt_graph_replay`). These are used ONLY by the pi05 CUDA-Graph provider. The llama_cpp provider supplies its own `step`/`run_stage` verbs and never enters that loop. The CLAUDE.md warning "stage semantics are CUDA-Graph-coupled" is therefore already half-obsolete: the v2 callback-stage seam decoupled it.

**Blast radius of a "vtable over exec/":** the seam already exists as free functions. Backend selection is LINK-TIME (one TU per build; `exec/CMakeLists.txt:24-34` hardcodes `cuda/cuda_backend.cpp`). To add a second exec backend you change ZERO files in `exec/src/` or `exec/include/`; you add one TU and one CMake line. The ONLY structural change the current design does NOT already accommodate is RUNTIME multi-backend selection in one process (linking both a CUDA and a CPU exec backend) — that would require promoting `frt::be::*` from free functions to a registered function-pointer table. No Phase 6 requirement demands this. Promoting it now is over-design.

## 3. Provider path needs + zero-copy export path analysis (lens 2)

### Jetson-PI provider TODAY (pure host-staged callback execution)

The provider never touches exec/. Verified: `cpp/providers/llama_cpp/src/pi0_runtime.cpp` includes only `flashrt/providers/llama_cpp/c_api.h` and `flashrt/model_runtime.h`. Every input is memcpy'd into provider-owned host buffers; every output is memcpy'd out.

- **Inputs (host memcpy IN):** images swizzled into host `std::vector<uint8_t> rgb_scratch`; prompt into `std::string`; state into `std::vector<float>`.
- **Infer:** allocates a local `std::vector<float> actions`, calls one opaque `jetson_pi_pi0_infer(...)` that writes into `actions.data()`, then assigns into `e->actions_buf`. GGML owns all device memory internally; FlashRT never sees a device pointer.
- **Output (host memcpy OUT):** `std::memcpy(out, e->actions_buf.data(), need_bytes)`.
- The `stream` parameter is dropped at every engine verb. All four ports are STAGED with null buffer / 0 offset / 0 bytes (`pi0_runtime.cpp:142-157`).

**What the provider needs from a backend abstraction: nothing.** Data volumes are batch=1 robot control (~300KB images, 32B state, 280B actions, 6.4KB diffusion noise). Host memcpy is negligible against multi-second (CPU) / hundreds-of-ms (Jetson GPU) diffusion compute. GGML's own internal per-step D2H recurrence (`llama-context.cpp:1254`) is a bigger serialization cost than any FlashRT-boundary copy would ever be. Zero-copy at the FlashRT boundary would not fix the in-provider serialization and is premature.

### CUDA-graph export zero-copy path (the de-facto same-backend memory-domain contract)

This path ALREADY works and is used end-to-end by pi05:

- **Buffer role is a bitmask** `FRT_RT_ROLE_INPUT|OUTPUT|STATE|SCRATCH` (`runtime.h:45-49`), with the comment "a buffer can be both input and output (e.g. an in-place diffusion noise/action buffer)."
- **Port SWAP window:** `frt_runtime_port_desc {buffer, offset, bytes}` (`model_runtime.h:154-156`). A non-null buffer turns the port into a raw device-window the host writes/reads directly. `frt_buffer_dptr(buffer)` (`exec.h:102`) returns the "stable device pointer"; `frt_buffer_wrap(ctx, name, void* dptr, bytes)` (`exec.h:101`) wraps an externally-owned device pointer (e.g. a torch tensor's `data_ptr`).
- **`frt_graph_bind`:** "Two graphs sharing one buffer on matching ports = zero-copy hand-off (this is the entire multi-subgraph / multi-model wiring mechanism)" (`exec.h:146`).
- **pi05 uses it:** `tensor_from_port` extracts `frt_buffer_dptr(p.buffer)` and tags `MemoryPlace::kDevice` (`model_runtime.cpp:74-96`), feeds it as `image_input_override`/`action_output_override` (`:414-415`); the `noise` port is `FRT_RT_PORT_SWAP` (`:335`).

**Is it abstract or CUDA-specific?** The HANDLE (`frt_buffer`) is opaque, but the SEMANTICS are CUDA-specific: `frt_buffer_dptr` returns a "stable device pointer," `frt_buffer_copy` is "device-to-device copy on a stream," streams/events are CUDA-shaped (`native_handle` is "raw backend stream, e.g. cudaStream_t" at `runtime.h:72-75`). It is NOT a multi-backend memory-domain abstraction; it is a CUDA-graph wiring mechanism that happens to use opaque handles. This is the canonical same-backend (CUDA) memory-domain contract and should be documented as such — it does not need to be generalized.

### Decision: do NOT build a device-buffer + sync abstraction for the provider-owned path

The CUDA export path already has one (CUDA-specific by construction); GGML must not be forced through it (would violate the no-GGML-types rule and the no-zero-copy-without-contract rule). The minimal, conservative Phase 6 deliverable is a memory-domain CONTRACT: a port annotation that lets a provider-owned port OPTIONALLY carry a buffer (opaque token) with honest ownership/lifetime/location/copy/sync, in place of today's forbidden SWAP window.

## 4. mllm reference patterns worth borrowing (lens 3)

mllm is reference architecture, NOT transplant material (CLAUDE.md). Worth borrowing for Phase 6:

1. **Device-tagged storage with tag-driven alloc/free routing** (`Storage.device_` + `MemoryManager::alloc` lookup). This is the minimal viable "memory domain" primitive: a buffer carries a domain tag, and a registry maps tag→allocator. FlashRT's analog: a `frt_memory_token` carries a `location_kind` tag; copy/sync dispatch by that tag.
2. **The three-collaborator split** (Backend / Allocator / Dispatcher as distinct objects). The analog for FlashRT: the `frt_memory_token` (memory) is distinct from the `executable` (create/replay) is distinct from the `provider` (model load/infer). mllm got this boundary right; Phase 6 should keep the token's verb set narrow (memory only), not let it grow into an execution vtable.
3. **Explicit `initXxxBackend()` registration into a singleton** rather than macro self-registration. For FlashRT with ~2 providers, manual registration is simpler and debuggable. Append-only and predictable. (NOT needed in Phase 6 — there is no second exec backend to register — but the pattern informs any future registry.)
4. **Copy-as-explicit-routed-op / honesty about cross-device transfer.** mllm's CPU X2X literally warns "should be implemented in device backends" (`X2XOp.cpp:13-15`). The right instinct: don't pretend zero-copy; route copies through the token's `copy_to_host`/`copy_from_host` verbs with explicit location_kind.

NOT worth borrowing (mismatch with FlashRT's low-latency CUDA-Graph focus):

- mllm's per-OpType op-factory table (60+ factories) — FlashRT providers execute whole graphs, not per-op.
- mllm's per-backend `Dispatcher` + `static_thread_pool` + stdexec senders — adds latency to a synchronous CUDA-Graph-replay hot path.
- mllm's `dlopen` `PluginOpPackageDescriptor` plugin system — FlashRT has two compile-time providers, not a plugin ecosystem.
- mllm's `BuddyMemPool` / large-tensor-threshold memory manager — FlashRT should use CUDA's native graph memory pools, not a framework buddy allocator.
- Any stream/event/executable-create/replay generality beyond what the CUDA-Graph provider concretely needs — mllm proves these are NOT naturally backend-neutral (each backend keeps them private).

Crucially: **mllm has NO stream/event abstraction, NO executable create/replay, NO host-callback, and NO memory-domain/import-handle concept in its backend-neutral interface.** Each backend manages streams/events privately; cross-device copy is a regular op, not a vtable method; there is no CUDA-Graph capture/replay analogue. This is direct evidence that "backend-neutral vtable for stream/event/executable" is NOT a naturally extractable abstraction — Phase 6 should not attempt it.

## 5. ABI/memory-domain constraints + the "vtable vs contract" decision (lens 4)

### ABI versioning story (already established, append-only)

FlashRT's ABI uses a two-field POD gate on every hand-off struct: `abi_version` + `struct_size`. Consumers gate with `>=` on struct_size (tail-extension probe):

- `frt_model_runtime_wrap` requires `exp->abi_version == FRT_RUNTIME_ABI_VERSION && exp->struct_size >= sizeof(frt_runtime_export_v1)` (`model_runtime.cpp:308-309`).
- `valid_model_runtime` requires `m->abi_version == FRT_MODEL_RUNTIME_ABI_VERSION && m->struct_size >= sizeof(frt_model_runtime_v1)` (`model_runtime.cpp:384-385`).
- `copy_verbs` reads v1 fields only if `verbs->struct_size >= sizeof(frt_model_runtime_verbs)` (`model_runtime.cpp:40`); `copy_verbs_v2` reads the v2 superset only if `verbs->struct_size >= sizeof(frt_model_runtime_verbs_v2)` (`:59`). Null entries are stub-filled with -3 unsupported (`:47-51, 67-72`).

This is the established pattern for adding vtable methods without breaking providers: a future `verbs_v3` with appended function pointers is readable by a v3-aware builder via the same `struct_size >=` probe; a v2 producer stays valid because its smaller `struct_size` satisfies the v2 reader. Phase 1-4 evolved strictly append-only — v2 structs alongside v1, never mutating v1, enums "ABI-frozen after v1 (append-only)" (`runtime.h:40`, `model_runtime.h:62`). Phase 6 MUST follow the same discipline.

### Phase 5 ABI gaps — triage against the memory-domain contract

The Phase 5 eval lists five gaps. Triaging each:

- **Gap 2 — provider-owned path forbids buffers and SWAP windows** (`model_runtime.cpp:203-217`; docs defer at `model_runtime_api.md:127-129`). THE Phase 6 core. Without lifting this, provider-owned ports cannot carry any buffer, so there is nothing for a memory-domain contract to annotate. **REQUIRED for Phase 6.**
- **Gap 1 — no intermediate/bidirectional port direction** (`model_runtime.h:101`, strict 2-value enum). Phase 5-revisit enabler for OBSERVABLE finer stages, NOT a memory-domain requirement. SEPARABLE. Phase 5's job.
- **Gap 3 — no partial/per-step/streaming output** (`model_runtime.h:201-203, 222-224`). Phase 5-revisit enabler. SEPARABLE.
- **Gap 4 — no stage-ready/stage-aware invalidation.** Phase 5-revisit enabler. SEPARABLE.
- **Gap 5 — monolithic engine vtable** (`c_api.h:96-115`, provider-internal, NOT FlashRT public ABI). The FlashRT public side ALREADY supports multi-stage via `run_stage(stage, stream)` and `add_callback_stage_v2` (a mixed graph+callback DAG is proven declarable). Gap 5 is a provider-internal split needing Jetson-PI sub-phase C entry points. SEPARABLE. Phase 5's job.

**Phase 6 minimal set = Gap 2 only.** Gaps 1, 3, 4, 5 are Phase 5-revisit enablers for OBSERVABLE finer stages; they do not block an honest memory-domain contract and should not be bundled into Phase 6 (scope creep with no Phase 6 consumer).

### The vtable-vs-memory-domain-contract decision

A **"backend vtable"** = alloc/free/stream/event/executable create/replay generalized to non-CUDA backends — the big abstraction CLAUDE.md Phase 6 lists as a *possibility*. This is the `exec.h` surface generalized. Building it would mean FlashRT gains a backend-registry/dispatcher layer akin to mllm's — which CLAUDE.md explicitly frames as "reference architecture, not transplant material."

A **"memory-domain contract"** = a small PORT ANNOTATION: an opaque FlashRT-owned token + ownership/lifetime/location/copy/sync rules, attached to a provider-owned port in place of today's forbidden SWAP window.

**Phase 6 should build the contract, NOT the vtable.** Three reasons:

1. **GGML is self-contained.** FlashRT does not run GGML ops; a FlashRT backend vtable would have no GGML ops to dispatch. The provider path's only FlashRT-visible need is letting a port OPTIONALLY carry a buffer for honest zero-copy or host readback — a port annotation, not an execution layer.
2. **Coexistence is already solved.** The v2 callback stage + provider-owned builder already lets CUDA-Graph and provider-owned runtimes coexist WITHOUT a shared backend abstraction. Phase 6 does not need to unify them.
3. **The only unblocking need is Gap 2.** Phase 5 explicitly defers the provider-owned SWAP rejection to Phase 6. Lifting it requires a contract that says "this provider-owned port may carry a buffer, here is how it is owned/located/copied/synced" — the memory-domain contract. Nothing in the Phase 5 gap list or the CLAUDE.md Phase 6 description requires FlashRT to gain alloc/free/stream/event/executable verbs for non-CUDA backends in this migration.

The temptation to build the full backend abstraction should be resisted: it adds surface area (new vtable, new registry, new dispatcher, new error paths) constrained by the no-GGML-types-in-public-ABI rule, with no consumer in this migration, and contradicts CLAUDE.md's "Jetson-PI as model provider, not bare exec/ hardware backend" and "mllm is reference architecture, not transplant material."

## 6. The memory-domain contract (ownership/lifetime/location/copy/sync; opaque-token vs device-pointer)

The contract must pin five facts (CLAUDE.md: "不要在没有 memory-domain contract 前假装可以零拷贝共享 buffer"):

1. **Ownership** — who allocates and who frees. The provider mints the token; FlashRT never allocates provider-owned backing store. The provider supplies a `destroy` verb; FlashRT calls it when the port/export refcount hits zero.
2. **Lifetime** — the token is valid only while the holder retains a reference (the export pattern at `runtime.h:138-145`). A provider-owned buffer's lifetime ties to the port/export retain-release, not to an implicit provider-internal pointer.
3. **Location** — host vs device, expressed as an OPAQUE FlashRT-defined `frt_rt_location_kind` enum (`HOST_VISIBLE`, `DEVICE_LOCAL`, ...), NOT a `ggml_backend_buffer_t` or `cudaMallocAsync` pointer. CLAUDE.md forbids leaking `ggml_tensor`/`ggml_cgraph`/`ggml_backend_t`/`ggml_backend_sched_t` into the public ABI.
4. **Copy semantics** — verbs `copy_to_host(dst, dst_off, src_off, bytes)` and `copy_from_host(src, src_off, dst_off, bytes)`. The existing CUDA `frt_buffer_copy` (`exec.h:110-111`) is the model; the token needs an equivalent implemented by the provider against its own backend.
5. **Sync point** — a `sync` verb (or "host-visible, no sync needed" declared via `location_kind`). Zero-copy becomes an ADVERTISED capability (`location_kind = HOST_VISIBLE`), NOT an assumption FlashRT makes.

**Opaque-token vs device-pointer: OPAQUE TOKEN.** Exposing a raw device pointer would (a) couple FlashRT to the provider's device-memory manager, (b) risk leaking GGML/CUDA specifics, (c) let a consumer dereference memory it does not own — violating the "no pretend zero-copy" rule. The existing `frt_buffer_dptr` accessor stays INTERNAL to the CUDA exec layer. For provider-owned ports, FlashRT sees only: token (opaque), bytes, location_kind, copy_to_host, copy_from_host, sync, destroy. The provider vouches for the buffer's contents and lifetime through these verbs; FlashRT never dereferences a provider-owned pointer.

## 7. How this layers over exec/ without rewriting it

The memory-domain contract lives at the **model-runtime / port layer**, NOT in exec/. Concretely:

- **exec/ is UNTOUCHED.** No changes to `exec/src/*.cpp` (`context.cpp`, `buffer.cpp`, `graph.cpp`, `plan.cpp`) or `exec/include/flashrt/exec.h`. The CUDA-graph export path (`frt_buffer` + `frt_graph_bind` + SWAP window + INPUT|OUTPUT role bitmask) remains the canonical same-backend (CUDA) memory-domain contract, documented as such.
- **backend.h gets a documentation split** (NOT a rewrite): clarify that the 13 free functions decompose into (a) shared memory/stream/event primitives (malloc/free, stream_*, event_*, memcpy_dtod_async, memset_async — every backend implements) and (b) CUDA-only graph-capture primitives (capture_begin/end, graph_exec_destroy, graph_launch — optional; a backend returning `false` from `capture_begin` simply cannot host graph-capture stages and must use the callback-stage path). GGML has no `cudaGraph_t` analogue; that subset is correctly CUDA-only and stays isolated in `cuda_backend.cpp`.
- **The contract is added at the model-runtime layer:** a new opaque `frt_memory_token` handle + a small `frt_memory_token_verbs` struct (copy_to_host, copy_from_host, sync, destroy) + a `frt_rt_location_kind` enum, plus an append-only extension to the port descriptor so a provider-owned port MAY carry a token in place of the today-forbidden `frt_buffer/offset/bytes` triple.
- **The provider-owned builder rejection (`model_runtime.cpp:209-217`) is relaxed for the token form ONLY.** A provider-owned port may carry a non-null `frt_memory_token` + offset/bytes + location_kind; it STILL rejects a raw `frt_buffer` (which would imply cross-backend device sharing that has no contract). The token is the ONLY allowed buffer form on provider-owned ports.

## 8. ABI evolution (append-only, versioning, what must not break)

- **New types are additive:** `frt_memory_token` (opaque typedef), `frt_memory_token_verbs` (new struct with its own `struct_size`), `frt_rt_location_kind` (new enum, values appended after existing enums — "ABI-frozen after v1 (append-only)").
- **Port descriptor extension is tail-only:** add a `frt_runtime_port_desc_v2` that EMBEDS the v1 layout and APPENDS `{frt_memory_token token; uint64_t offset; uint64_t bytes; uint32_t location_kind; uint32_t reserved}`. A v1 producer's smaller `struct_size` continues to satisfy the v1 reader; a v2-aware builder reads the trailing fields only when `struct_size >= sizeof(frt_runtime_port_desc_v2)` — the identical pattern `copy_verbs`/`copy_verbs_v2` already use. Old consumers and old providers keep working unchanged.
- **Builder API is appended, not mutated:** add `frt_runtime_builder_add_port_v2` (or an overload) accepting the token form; the existing `frt_runtime_builder_add_port` is untouched. The v2 `finish_model_v2` path's provider-owned rejection (`model_runtime.cpp:203-217`) is RELAXED in-place to permit the token form while still rejecting raw `frt_buffer` — this is a behavioral relaxation on an existing v2 path, not a struct-layout change, so v2 binaries are not broken (a v2 producer that supplies no token sees identical behavior).
- **Token verbs follow the stub-fill discipline:** absent verbs are replaced by unsupported stubs returning -3 (`model_runtime.cpp:47-51, 67-72`), so every entry is always callable.
- **What MUST NOT break:** (a) every Phase 1-4 provider binary that fills `frt_model_runtime_verbs`/`verbs_v2` with v1/v2-sized structs; (b) `frt_model_runtime_wrap`'s `abi_version`+`struct_size` gate (`model_runtime.cpp:308-309`); (c) `valid_model_runtime`'s gate (`:384-385`); (d) the CUDA-graph export path's `frt_buffer`/`frt_graph_bind`/SWAP mechanism (byte-identical). The memory-domain contract is a NEW, parallel surface; it does not alter the CUDA export path's contract.

## 9. In-scope (minimal) vs out-of-scope (deferred)

See the structured `in_scope` / `out_of_scope` fields. The boundary is: Phase 6 delivers the memory-domain CONTRACT (token + verbs + port extension + builder relaxation + backend.h doc split + smoke test + documentation). It does NOT deliver a full exec/backend vtable, a runtime-selectable backend registry, cross-backend zero-copy implementation, or any of the Phase 5-revisit ABI gaps (1, 3, 4, 5).

## 10. Implementation plan (ordered)

See the structured `implementation_plan` field. The order is: (1) document the two existing contracts first (no code risk, clarifies the boundary); (2) add the token types + location_kind enum (purely additive headers); (3) add the port descriptor v2 + builder entry (append-only); (4) relax the provider-owned rejection for the token form only (behavioral, gated); (5) add the token verbs stub-fill + lifetime wiring to retain/release; (6) backend.h doc split + optional `import` declaration (no exec/src changes); (7) smoke test (provider-owned runtime with a host-visible token, round-trip copy, verify v1 consumers unchanged).

## 11. Risks + mitigations

See the structured `risks` field. Headline risks: (a) scope creep into a backend vtable — mitigated by explicit out-of-scope list and CLAUDE.md constraints; (b) token verb set too narrow for future needs — mitigated by the struct_size append-only probe (verbs_v2/v3); (c) provider mis-advertises location_kind — mitigated by FlashRT never dereferencing the token (only copy verbs); (d) lifetime mismatch — mitigated by tying destroy to port/export retain-release; (e) Phase 5 revisit needs MORE than the token — acknowledged, Phase 6 satisfies condition (e) only, not (a)-(d).

## 12. Phase 5 coupling — does this satisfy revisit condition (e)?

**Yes, this satisfies Phase 5 revisit condition (e) — and ONLY (e).** Phase 5's condition (e) is an OR: "Phase 6 memory-domain contract exists (or new append-only ABI: intermediate port direction, per-step `get_output`, stage-ready flag)." The memory-domain contract is the first disjunct. By delivering the token + port extension + builder relaxation, Phase 6 removes the FlashRT-side ABI blocker (`model_runtime.cpp:209-217`, the Gap 2 deferral) that today prevents provider-owned ports from carrying any buffer.

**Important honesty caveat:** satisfying (e) does NOT by itself make finer Pi0 stages worth doing. Conditions (a)-(d) are Jetson-PI-INTERNAL: (a) on-device recurrence, (b) graph reuse across the 10 denoise steps, (c) cross-tick double-buffered encoder KV, (d) profiling showing the encoder is a non-trivial hideable fraction. None of those are FlashRT's concern, and none are unblocked by Phase 6. The Phase 5 eval's conclusion stands: even when (a)-(e) ALL hold, the correct implementation is a **provider-internal pipeline exposed to FlashRT as ONE callback stage** — not a FlashRT stage decomposition. Phase 6's memory-domain contract is the necessary-but-not-sufficient FlashRT-side enabler; it lets a FUTURE provider-internal pipeline (should one materialize) optionally expose honest host-visible or device-local buffers through provider-owned ports WITHOUT forcing the provider to fake a CUDA graph or leak GGML types. That is the smallest honest contribution Phase 6 can make to the Phase 5 revisit, and it is exactly the contribution CLAUDE.md Phase 6 asks for ("在 contract 明确后再考虑 zero-copy").

---

## 13. Phase 6 decision (recorded 2026-07-07)

**Decision: GO-MINIMAL-CONTRACT.** Build the small, append-only memory-domain
contract; do NOT build a full backend-neutral exec/backend vtable in this phase.

The four lenses converge: exec/ already has a backend-neutral *shape* (13 free
functions, CUDA isolated to one 107-line TU, zero `virtual`); the provider-owned
callback-stage bypass already exists and is in production use by the llama_cpp
provider (which never includes exec.h); GGML is self-contained so a full
backend vtable would have no consumer; the ONLY genuinely missing piece for the
stated Phase 6 zero-copy goal is the memory-domain CONTRACT (Gap 2, explicitly
deferred to Phase 6 at docs/model_runtime_api.md:127-129). Building a full
backend vtable now would add surface area with no consumer and contradict
CLAUDE.md ("Jetson-PI as model provider, not bare exec/ hardware backend";
"mllm is reference architecture, not transplant material").

### The memory-domain contract (opaque token, NOT raw device pointer)
- Ownership: provider mints `frt_memory_token` + supplies `destroy`; FlashRT
  never allocates provider-owned backing store; `destroy` called once when
  port/export refcount hits zero.
- Lifetime: token valid only while holder retains a reference (export pattern
  runtime.h:138-145).
- Location: opaque `frt_rt_location_kind` enum {HOST_VISIBLE, DEVICE_LOCAL} --
  never a ggml_backend_buffer_t / cudaMallocAsync pointer.
- Copy: `copy_to_host` / `copy_from_host` verbs (model: exec.h frt_buffer_copy).
- Sync: `sync` verb, or "host-visible, no sync" declared via location_kind.
- Zero-copy becomes an ADVERTISED capability (location_kind=HOST_VISIBLE), not
  an assumption FlashRT makes. FlashRT NEVER dereferences the token.

### Two coexisting contracts
- (A) CUDA-export path: opaque `frt_buffer` + `frt_buffer_dptr`/`wrap` +
  `frt_graph_bind` + port SWAP window + INPUT|OUTPUT role bitmask. CUDA-specific
  semantics, canonical same-backend hand-off. UNTOUCHED by Phase 6.
- (B) Provider-owned token path: the new opaque-token contract above, the only
  buffer form permitted on provider-owned ports.

### ABI evolution (strictly append-only; Phase 1-4 binaries unchanged)
- New enum `frt_rt_location_kind` (values appended; ABI-frozen-after-v1 respected).
- Port descriptor extension via the established `struct_size >=` tail probe
  (identical to copy_verbs / copy_verbs_v2 at model_runtime.cpp:40,59).
- Token verbs struct carries its own `struct_size`; absent verbs stub-filled -3.
- Provider-owned builder rejection (model_runtime.cpp:209-217) relaxed for the
  token form ONLY; raw `frt_buffer` rejection stays.
- What MUST NOT break: abi_version+struct_size gates (model_runtime.cpp:308-309,
  384-385); every Phase 1-4 provider binary; the CUDA-export path byte-identical.

### Phase 5 coupling
Satisfies Phase 5 revisit condition (e) -- and ONLY (e). Conditions (a)-(d) are
Jetson-PI-internal (on-device recurrence, graph reuse, double-buffered KV,
profiling). The token is the necessary-but-not-sufficient FlashRT-side enabler;
Phase 5 ABI gaps 1/3/4/5 (intermediate port direction, per-step get_output,
stage-ready flag, engine run_stage-by-index) stay OUT of Phase 6 scope.

### In-scope deliverables (minimal)
1. Document the two existing contracts (no code risk).
2. Add token types + `frt_rt_location_kind` enum (additive headers).
3. Port descriptor v2 + builder entry (append-only).
4. Relax provider-owned rejection for token form only (gated behavioral).
5. Token verbs stub-fill + lifetime wiring to retain/release.
6. backend.h documentation split (shared primitives vs CUDA-only graph-capture);
   declare `frt::be::import` extension point (not implemented for non-CUDA).
7. Smoke test: provider-owned runtime with a host-visible token, round-trip copy,
   v1-consumer unchanged, raw frt_buffer still rejected.

### Explicitly out of scope (over-design / Phase 5-revisit territory)
- Full exec/backend vtable (alloc/free/stream/event/executable) for non-CUDA.
- Runtime-selectable backend registry (link-time selection already suffices).
- Cross-backend zero-copy implementation (batch=1 robot control is premature).
- Phase 5 ABI gaps 1/3/4/5.
- Any exec/src/ struct refactor; mllm op-factory / dispatcher / plugin patterns.
