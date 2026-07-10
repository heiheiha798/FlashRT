# Phase 5 Evaluation — Finer Pi0 Stages (context / encoder / diffusion / action)

> Implementation update (2026-07-10): the earlier no-go below is superseded at
> the stable boundary required by the complete migration. Jetson-PI now exposes
> separate `prepare_context` and `run_action` narrow C API calls, and FlashRT
> exposes callback stages `context -> action` while preserving `infer` as the
> compatibility whole-tick face. The context stage owns image preprocessing,
> VIT, language/image encode, and provider-private encoded cross-KV; action owns
> the complete ten-step diffusion loop and consumes the pending context once.
> Whole infer versus split execution is bit-identical (`max_abs_diff=0`) on
> CUDA and SYCL. This does not claim intermediate diffusion actions or an
> intra-tick scheduling speedup; the historical analysis remains authoritative
> for any split finer than the stable context/action boundary.

> Historical evaluation dated 2026-07-07. The implementation update above is
> authoritative; the remaining text records the original performance gate.

## 1. Executive summary + recommendation

**Recommendation: NO-GO. Defer Phase 5.**

Four independent lenses (Pi0 internal compute structure, FlashRT provider surface, FlashRT ABI expressiveness, scheduling-benefit) converge on the same conclusion: splitting Pi0 into finer FlashRT-visible stages yields **zero latency win under current conditions** and adds ABI surface, new ports, new sync points, and new error paths for no return. The dominant optimizations live *inside* Jetson-PI's diffusion phase (Phase C), not at a new FlashRT stage boundary.

The single most important structural fact: Pi0 is **flow-matching diffusion with atomic chunk emission**, not autoregressive generation. The K=10 denoise steps refine a *latent* `[50, action_dim]` in lockstep; `cross.action_ready` is set `true` only after the full loop completes (`Jetson-PI/src/llama-context.cpp:1389`), and `llama_get_pi0_action` refuses early reads (`:798`). Intermediate latents are not valid actions (each velocity contribution is scaled by `1/n_inference_steps`, `Jetson-PI/src/models/pi0_ae.cpp:109`). **Time-to-first-action == time-to-full-chunk.** There is no streaming win to capture, which removes the most attractive reason to split.

The second structural fact: the encoder (Phase B) already runs **once per tick, outside** the K-loop (`llama-context.cpp:975` encode, then `:1241` K decode calls). The encoder→diffusion dependency is already serialized once and non-redundant. Splitting B and C into two FlashRT stages cannot overlap them within a tick because diffusion's first step consumes encoder cross-KV (`pi0_ae.cpp:13,68-69`). The only theoretical scheduling win is *cross-tick* pipelining (tick N+1's VIT+encoder behind tick N's K diffusion), and that is a provider-internal streaming change on separate GGML streams with double-buffered encoder KV — **not a FlashRT stage decomposition**.

The Phase 5 gate in CLAUDE.md requires all four of: scheduling benefit, buffer contract, sync semantics, profiling evidence. On current evidence **none of the first three pass**, and the fourth (profiling) is to be backfilled by the main process using the plan in this document. The profiling plan's purpose is to *confirm* the no-go (diffusion dominates, graph-reuse is the real lever) and to set the numeric bar any future revisit must clear — not to search for a justification to split.

A no-go here is the correct, defensible outcome: it prevents adding ABI surface and stage-boundary overhead for zero latency gain on the actual Jetson deployment, and it correctly routes the real optimization work (graph reuse across the 10 denoise steps; eliminate the per-step D2H recurrence; eliminate the cross-KV host round-trip) to where it belongs — inside Jetson-PI, behind the existing single `infer` stage.

---

## 2. Pi0 internal phase map (one policy tick)

Entry: `jetson_pi_pi0_infer` (`Jetson-PI/src/jetson_pi_pi0.cpp:205-344`). The whole compute is delegated to `mtmd_helper_eval_chunks_pi0` (`Jetson-PI/tools/mtmd/mtmd-helper.cpp:446-794`); only the final action readout is a separate C-API call. Five phases:

| Phase | What | Where | Cost shape |
|---|---|---|---|
| A — VIT image encoder | Per-view CLIP-VIT pass; output `ctx->image_embd_v` `[n_img_tokens=196, n_mmproj_embd=1152]` F32 per view | `mtmd-helper.cpp:575` → `mtmd_encode` → `clip_image_batch_encode` (`tools/mtmd/mtmd.cpp:780,809,835`); graph `clip_ctx::build_pi0` (`tools/mtmd/clip.cpp:626-653`) | 1 pass per view (LIBERO: 2 views). Already timed ("Vit took"). |
| B — Language+image context encoder | Single non-causal prefill over all image+text tokens; first half of 27 layers (il=0..12). NO KV cache (`build_attn_inp_no_cache_pi0`). | `llama_encode` at `mtmd-helper.cpp:766` → `llama_context::encode` (`llama-context.cpp:911`) → `process_ubatch(LLM_GRAPH_TYPE_ENCODER)` (`:975`); graph `llm_build_pi0` (`src/models/pi0.cpp:4-151`), dispatched `llama-model.cpp:7180-7182`. | 1× per tick. Side outputs: `cross.encoded_kv_data[i]` (per-layer cross-KV) + `cross.action` (Gaussian noise latent `[50,32]`) + `cross.state`. |
| C — Diffusion flow-matching denoise loop | K=10 Euler steps; each builds+computes the AE decoder graph (second half of layers, il=13..26) over `action_steps+1=51` query tokens with cross-attention to encoder KV. | Loop `for (i=0; i<inference_steps; i++)` at `llama-context.cpp:1241`; per-step `process_ubatch(LLM_GRAPH_TYPE_DECODER)` at `:1245`; graph `llm_build_pi0_ae` (`src/models/pi0_ae.cpp:4-117`), dispatched `llama-model.cpp:7184-7185`. | K=10 (verified from GGUF `pi0.num_inference_steps=10`). **Critical path.** |
| D — action commit | `cross.action_ready = true` after loop. | `llama-context.cpp:1388-1389`. | negligible |
| E — action readout | Pure `memcpy` of `cross.action` `[50,32]`. | `jetson_pi_pi0.cpp:328` → `llama_context::get_pi0_action` (`llama-context.cpp:788-806`). | negligible (memcpy) |

**Hyperparameters verified from the GGUF** (`pi0.num_inference_steps=10`, `pi0.n_action_steps=50`, `pi0.max_action_dim=32`, `pi0.block_count=27`, `pi0.embedding_length_ae=1024`), bound at `Jetson-PI/src/llama-model.cpp:1244-1246`. LIBERO effective action_dim=7 (from `stats.json`); `max_action_dim=32` is the model's padded action dim, sliced/denormalized downstream.

**Diffusion loop internals (the hot path):**
- `time_step` schedule: `cross.time_step = float(1 - i/float(inference_steps))` (`llama-context.cpp:1242`) → t = 1.0, 0.9, …, 0.1.
- Per step: `process_ubatch(DECODER)` builds+computes one velocity prediction `v_theta(x_t, t)` (`pi0_ae.cpp:113`, shape `[action_dim*(action_steps+1)]`).
- Per-step update is an **Euler step in HOST code**: `res->get_action()` is copied device→host via `ggml_backend_tensor_get_async` (`llama-context.cpp:1254-1260`), then `cross.action[j] -= action_data[action_dim+j]` (`:1263-1264`). The `1/n_inference_steps` scale is baked into the graph at `pi0_ae.cpp:109`.
- **State carried between steps: ONLY `cross.action`** (updated in place). `cross.encoded_kv_data` and `cross.state` are CONSTANT across all 10 steps. No KV cache, no recurrent state.
- **The decoder graph is REBUILT + reallocated on every one of the 10 steps.** `can_reuse` returns `false` for `llm_graph_input_state` (`llama-graph.cpp:109`), `llm_graph_input_action` (`:137`), `llm_graph_input_sinusoidal_embedding` (`:192`), and `llm_graph_input_attn_no_cache_ae` inherits the base default `return false` (`llama-graph.h:100-105`, no override). So `process_ubatch` falls into the rebuild branch (`llama-context.cpp:864-887`) 10× per tick. This is the single most actionable inefficiency, and it is a Jetson-PI fix — not a FlashRT stage split.

**Phase-boundary buffers (the contract surface a split would have to specify):**

| Boundary | Buffer | Shape | Lives in | file:line |
|---|---|---|---|---|
| A→B (per view) | `ctx->image_embd_v` | `[196, 1152]` F32 | mtmd ctx host | `mtmd.cpp:817`, `mtmd-helper.cpp:589` |
| B→C (encoder→decoder) | `cross.encoded_kv_data[i]`, i=0..n_layer-1 (per-layer K and V) | each `[n_embd_head=256, n_head_kv=16, n_enc]` F32, n_enc=n_tokens | host (D2H via `ggml_backend_tensor_get_async`) | `llama-graph.h:75`, `llama-context.cpp:1066-1071`, `llama-graph.cpp:441-457` |
| B→C | `cross.action` (noise latent x_T) | `[50, 32]` F32 | host | `llama-context.cpp:1077` |
| B→C | `cross.state` (proprioception) | `[32]` F32 (padded) | host | `llama-context.cpp:1079-1081`, `llama-graph.cpp:102-107` |
| C internal (per step) | `res->action` (velocity v_theta) | `[32*(50+1)]` F32 | GPU tensor | `pi0_ae.cpp:113`, read `llama-context.cpp:1251` |
| C→D / D→E | `cross.action` (final x_0) | `[50, 32]` F32 | host | `llama-context.cpp:1264,1389,804` |
| cross-step constant | `cross.encoded_kv_data`, `cross.state` | as above | host | unchanged across 10 steps |

The B→C cross-KV is the heaviest contract: n_layer·2 tensors, currently materialized host-side via an explicit D2H with a `// TODO: tmp` hack comment (`llama-graph.h:56-60`). This device→host→device round-trip (`llama-context.cpp:1069` D2H, `llama-graph.cpp:455` H2D feed-back per denoise step) is itself a serialization cost and a known buffer-contract sore point.

**Streaming potential: effectively NONE.** (1) The chunk is produced all-at-once at the END (`action_ready` only after the loop, `llama-context.cpp:1389`; early reads rejected `:798`). (2) Even intermediate `cross.action` after each Euler step is the latent at time t_i, NOT a usable action — it must complete all 10 steps to converge to x_0. The per-step velocity is a gradient, not an action. There is no "early action step available before later ones" semantics; the 50 action timesteps emit together as one `[50,32]` block. The only streaming-shaped opportunity is running fewer denoise steps (K'<10) for a lower-quality chunk — that is a real per-tick latency reduction, but it trades quality for latency and is orthogonal to stage splitting: the whole-graph path already exposes it via `pi0.num_inference_steps` in the GGUF, no stage boundary needed. Stage splitting does not unlock it.

**Cross-tick context reuse: zero.** The Pi0 encoder is stateless/no-cache (`build_attn_inp_no_cache_pi0`), KV is cleared every tick (`jetson_pi_pi0.cpp:234`), and the camera images change every tick in a robot deployment — so there is no encoder output to reuse across ticks even when the prompt is unchanged. A `prefill_context` stage split therefore cannot amortize across ticks.

---

## 3. Current whole-graph path summary (FlashRT provider surface)

The provider exposes **four STAGED ports and exactly ONE callback stage**:

- Ports (`FlashRT/cpp/providers/llama_cpp/src/pi0_runtime.cpp:142-157`): `images` IN (IMAGE/U8/NHWC `[n_views,H,W,C]`), `prompt` IN (TEXT/U8/FLAT `{-1}`), `state` IN (STATE/F32/FLAT `[action_dim]`), `actions` OUT (ACTION/F32/FLAT `[action_steps,action_dim]`). All STAGED, none with a SWAP device window (buffer=nullptr, offset=0, bytes=0).
- Stage: `frt_runtime_builder_add_callback_stage_v2(b, "infer", 0, nullptr, 0)` (`pi0_runtime.cpp:158-159`). Stage index enum frozen at `FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER=0` (`c_api.h:25-27`).
- `run_stage` (`pi0_runtime.cpp:58-73`) accepts ONLY `INFER` (`:62`) and forwards to `owner->engine.run_infer` (`:66`); any other index returns "unknown stage" (`:63`). `step` (`:75-77`) is sugar calling `run_stage(INFER,-1)`.
- Engine vtable `frt_llama_cpp_engine_v1` (`c_api.h:96-115`) has only `set_input` / `run_infer` / `get_output` — a SINGLE monolithic infer verb. `run_infer` (`jetson_pi_engine.cpp:229-255`) gates on `images_set/prompt_set/state_set` (`:233`), allocates a local `actions` vector (`:239-240`), calls **one opaque blocking C call** `jetson_pi_pi0_infer(...)` (`:242-247`), and copies the result into `e->actions_buf` (`:253`). The `stream` parameter is dropped at every engine verb (`jetson_pi_engine.cpp:151,265`).
- `get_output` (`jetson_pi_engine.cpp:264-291`) serves only ACTIONS, requires whole-chunk capacity (`:277-284`), gates on `actions_buf.size()==need_elems` (`:285`, else `-7 ACTION_NOT_READY`), and memcpys the whole chunk (`:289`).
- Invalidation is blunt: `e->actions_buf.clear()` on **every** `set_input` regardless of port (`jetson_pi_engine.cpp:157`).

**The only FlashRT-visible seam is the single `jetson_pi_pi0_infer` call at `jetson_pi_engine.cpp:242-247`.** Everything inside that call (VIT, context prefill, KV fill, diffusion loop, action decode) is opaque to FlashRT. A split into `[encode_images, prefill_context, diffusion_loop, decode_actions]` therefore requires **Jetson-PI to expose separate sub-phase C entry points**; FlashRT cannot fabricate the boundary from its side. Profiling the sub-phases from the FlashRT side is impossible for the same reason — the only FlashRT-side instrumentation point is bracketing `jetson_pi_pi0_infer`.

---

## 4. FlashRT ABI capability assessment

**Can the ABI DECLARE finer stages today? YES — append-only, no breakage.** `frt_runtime_stage_desc_v2` (`FlashRT/runtime/include/flashrt/model_runtime.h:172-179`) carries `{name, kind, graph, callback, n_after, after[]}`; `frt_runtime_builder_add_callback_stage_v2` (`:323-327`) appends one stage per call; `run_stage(self, stage, stream)` (`:232`) fires one stage by index. A mixed graph+callback 2-stage DAG is proven by `test_model_runtime.cpp:394-409`. So 4 callback stages with `after` edges are declarable today. The single-stage Pi0 provider is a provider choice, not an ABI ceiling.

**Can the ABI host OBSERVABLE finer stages with inter-stage buffer contracts today? NO.** Four concrete gaps:

1. **No intermediate port direction.** `frt_rt_port_direction` is a strict 2-value enum, NOT a bitmask: `FRT_RT_PORT_IN=0 / FRT_RT_PORT_OUT=1` (`model_runtime.h:101`). A port cannot be stage-N-out AND stage-(N+1)-in. STAGED ports only flow caller→engine (IN via `set_input`) and engine→caller (OUT via `get_output`). Every inter-stage tensor (image embeddings, encoder KV, diffusion latent, action logits) must therefore be a PRIVATE provider buffer; FlashRT cannot mediate, fingerprint, or sync it.

2. **Provider-owned path forbids buffers and zero-copy windows.** The builder rejects all streams/graphs/buffers/regions on provider-owned runtimes (`FlashRT/runtime/src/model_runtime.cpp:203-208`) and rejects any SWAP window / offset / bytes on ports (`:209-217`). The export-level buffer `role` IS a bitmask allowing INPUT|OUTPUT (`runtime.h:45-49`, comment "an in-place diffusion noise/action buffer") and `frt_graph_bind` enables zero-copy hand-off between CUDA graphs sharing one buffer — but that mechanism belongs to the **CUDA-graph export path only**, not provider-owned v2. The docs state this explicitly: "Provider-owned v2 ports are STAGED only in this first contract. The builder rejects raw SWAP windows on this path until FlashRT has an explicit memory-domain contract for cross-provider buffers" (`docs/model_runtime_api.md:127-129`). That contract is the **Phase 6 deferral**.

3. **No partial / per-step / streaming output.** `get_output(self, port, out, capacity, written, stream)` (`model_runtime.h:201-203,222-224`) is a full-snapshot read of ONE port by index. No step-k/K parameter, no per-step ready flag, no streaming token. If diffusion were split into K denoise sub-stages, each would be an opaque `run_stage` call with NO observable intermediate output until the final decode. The Pi0 engine confirms the all-or-nothing shape: `set_input` clears `actions_buf` (`jetson_pi_engine.cpp:157`), `get_output` returns the whole chunk only after `run_infer` completes (`:264-291`).

4. **Engine vtable is monolithic.** `frt_llama_cpp_engine_v1` (`c_api.h:96-115`) exposes only `run_infer`. `jetson_pi_pi0_infer` (`jetson_pi_engine.cpp:242-247`) is one opaque call. Backing 4 stages needs either a generic `run_stage(stage_idx)` engine verb or per-sub-stage verbs, AND Jetson-PI must expose the sub-phase C entry points. Both are provider-internal changes.

**Implication.** Finer stages can be prototyped TODAY only as **provider-internal sub-steps hidden behind the existing single `infer` stage** (the Phase 1 whole-graph shape stays the externally declared surface). Promoting them to first-class FlashRT-observable stages — where FlashRT mediates/fingerprints/syncs the buffers between encode→prefill→diffusion→decode and the host can read partial action progress — requires Phase 6-style work first (memory-domain/SWAP extension to provider-owned ports, or new append-only ABI concepts: intermediate port direction, per-step `get_output`, stage-ready flag). The current ABI alone cannot host observable finer Pi0 stages without new abstractions.

---

## 5. Scheduling-benefit analysis (with honest counterargument)

### 5.1 Action-chunk streaming — NO win
Pi0 is flow-matching diffusion, not autoregressive. The K steps refine a latent in lockstep; the chunk is emitted atomically after the loop. Time-to-first-action == time-to-full-chunk. **Theoretical latency reduction from action streaming: 0.** (See phase map §2.)

### 5.2 Prefill/diffusion overlap — within-tick blocked; cross-tick bounded
Within one tick: VIT → encoder (13 layers, non-causal, once) → K-step diffusion (14 layers, K=10). The encoder already runs **once, outside** the K-loop (`llama-context.cpp:975` then `:1241`). Diffusion's first step consumes encoder cross-KV, so encoder→diffusion is a strict prerequisite — but only once per tick, not once per K step. The architectural seam already exists; there is no *redundant* work to remove by splitting B and C into two FlashRT stages.

Cross-tick pipelining (tick N+1's VIT+encoder overlapping tick N's K diffusion) is the **only genuine scheduling opportunity**. On a single Jetson GPU it requires: (a) encoder and decoder on independent streams with no shared backend-scheduler mutex; (b) double-buffered encoder KV so tick N's diffusion reads buffer A while tick N+1's encoder writes buffer B; (c) **elimination of the per-step D2H+host-recurrence serialization barrier** (`llama-context.cpp:1254` D2H of `res->action`, then `:1263-1264` host Euler update) so the diffusion loop is GPU-resident. None of (a)(b)(c) exist today. Even with all three, the win is bounded by `min(encoder_time, K×step_time)` and lives entirely inside the provider — it is not a FlashRT stage decomposition. The correct exposure even then is ONE callback stage wrapping a provider-internal pipeline.

### 5.3 Batching — NO dimension on a single Jetson
`cparams.n_seq_max=1` (`jetson_pi_pi0.cpp:135`), KV cleared every tick (`jetson_pi_pi0.cpp:234`), explicitly stateless single-tick. Batch=1 is the deployment norm on one Jetson controlling one robot. There is no second concurrent tick to batch. Finer stages do not unlock batching on the target deployment.

(VIT *cross-view* batching — combining LIBERO's 2 camera views into one VIT pass — is a potential latency win, but it is an mtmd/clip-internal change to the per-view loop at `mtmd-helper.cpp:575`; a FlashRT `encode_images` stage split would still call the same per-view mtmd path and gain nothing. It is orthogonal to staging and is a Jetson-PI-internal optimization.)

### 5.4 Critical path — diffusion dominates; only a future tick's encoder is hideable
Tick critical path: VIT → encoder (1×) → K× decoder (10×). The critical phase is the K-step diffusion (10× decoder passes + 10× per-step D2H+host recurrence syncs). The only non-critical, hideable phase is a *future* tick's VIT+encoder — i.e. cross-tick pipelining (§5.2), not intra-tick stage splitting. Within a single tick, encoder→diffusion is already serial and non-redundant, so splitting it adds a boundary with zero overlap benefit.

### 5.5 Honest counterargument (strongest case AGAINST splitting)
1. **Sequential data dependency in diffusion**: step i+1's input `cross.action` is step i's output (`llama-graph.cpp:133` uploads `cross->action.data()`; `llama-context.cpp:1264` updates it). The K steps are a strict serial chain. No intra-tick parallelism across K.
2. **No incremental action emission**: `action_ready` set once after all K steps (`llama-context.cpp:1389`); intermediate latents are not valid actions. Zero time-to-first-action win.
3. **Encoder already amortized once per tick**: not inside the K-loop. Splitting encoder/diffusion into two FlashRT stages cannot overlap them within a tick — diffusion depends on encoder KV.
4. **The whole-graph path already amortizes graph build in principle** (`llm_graph_result::can_reuse`, `llama-graph.cpp:813`), BUT `llm_graph_input_action::can_reuse` returns `false` (`llama-graph.cpp:137-139`), forcing a rebuild every K step today. **This is a Jetson-PI-side inefficiency to fix, NOT a reason to add FlashRT stage boundaries.** Adding stages does not fix it; fixing `can_reuse` (or pinning the action input as a `set_input` rather than a rebuild trigger) does.
5. **Stage-boundary overhead + ABI complexity for no win**: each new FlashRT stage = new ports, new staged-buffer contracts, new sync points, new error paths, all constrained by CLAUDE.md's no-GGML-types-in-public-ABI rule. The single `infer` callback stage is the minimal correct surface. Splitting adds surface area proportional to stage count with zero latency return unless cross-tick pipelining is ALSO built — and that pipelining lives inside the provider, not in FlashRT's stage graph.
6. **The per-step D2H+host recurrence** (`llama-context.cpp:1254,1263`) is the real serialization bug, independent of staging. Fix it (move recurrence on-device, keep `cross.action` resident) before considering any stage split.
7. **Multi-callback middle-ground considered and rejected**: one could declare multiple callback stages sharing provider-internal state via the engine `self` pointer (no observable inter-stage port needed, no new ABI) — e.g. `encode_images` → `prefill` → `diffusion` → `decode` as separate `run_stage` calls on one engine. This IS expressible today. But it creates no overlap (the intra-tick data dependencies above still force strict serial execution), and the only thing it buys is host-side insertion points between phases (telemetry, early abort). Those insertions carry no latency win and add stage-boundary dispatch overhead. It is therefore cosmetic decomposition under current conditions, not a scheduling win.

**Splitting would NOT help when**: batch=1 (the Jetson norm), single-robot single-tick latency is the objective, cross-tick pipelining is not built, and the per-step D2H sync remains. These are exactly the current conditions. Under them, finer stages are cosmetic decomposition.

---

## 6. The four gates assessed individually

CLAUDE.md Phase 5: "拆分前必须明确调度收益、buffer contract、同步语义和 profiling 证据."

### Gate 1 — Scheduling benefit: **FAIL**
No intra-tick benefit. K diffusion steps are a strict serial chain. Encoder already runs once outside the loop. The only theoretical win (cross-tick encoder/diffusion pipelining) requires provider-internal work (separate streams, double-buffered KV, on-device recurrence) that is not a FlashRT stage split. Action streaming is a hard NO (atomic chunk emission). Batching is a NO on a single Jetson.

### Gate 2 — Buffer contract: **FAIL (gated on Jetson-PI + Phase 6)**
The B→C boundary passes n_layer cross-KV tensors currently forced through a device→host→device round-trip with an explicit TODO hack (`llama-graph.h:56-60`). Any split placing B and C in different memory domains must specify ownership/lifetime of these tensors; today the host copy is the de-facto contract. A|B and C|D/E are trivial host float buffers and do not block. But the harder problem is on the FlashRT side: the provider-owned ABI path forbids inter-stage buffers and SWAP windows (`model_runtime.cpp:203-217`), so observable inter-stage contracts cannot be expressed without Phase 6 memory-domain work.

### Gate 3 — Sync semantics: **FAIL**
The per-step `ggml_backend_tensor_get_async` D2H of the action tensor (`llama-context.cpp:1254`) followed by the host-side Euler update (`:1263-1264`) is itself a serialization barrier that must be eliminated before ANY overlap — intra- or cross-tick — can materialize. This is a Jetson-PI fix, independent of staging. On the FlashRT side, the only readiness signal is `actions_buf.size()==need_elems` (`jetson_pi_engine.cpp:285`); invalidation is the blunt `clear()` on any `set_input` (`:157`). A multi-stage split would need per-stage readiness flags and stage-aware invalidation, neither of which exists, and each new sync point can erase the scheduling benefit if the stage is short.

### Gate 4 — Profiling evidence: **NOT YET COLLECTED — backfill plan in §8**
The code has printf instrumentation (Vit/Encode/Decode at `mtmd-helper.cpp:579,781-783`) but the per-step `step_decode_ms` computed at `llama-context.cpp:1270` is **NOT printed** (verified: only `printf("inference_time: %d over\n",i)` at `:1247` and a separator at `:1248`). No encoder-vs-diffusion ratio is reported. The profiling-evidence requirement is unmet; the plan in §8 is what the main process executes to backfill it. **The structural analysis is strong enough to recommend no-go now; profiling is to confirm and to set the numeric bar for any future revisit, not to search for a justification.**

---

## 7. What would change the answer (conditions for revisiting)

A real, non-cosmetic scheduling benefit exists **only if ALL of these hold**:

- **(a) On-device recurrence**: the per-step D2H action copy + host Euler update (`llama-context.cpp:1254,1263`) is eliminated so `cross.action` stays GPU-resident and the diffusion loop has no per-step host sync. (Jetson-PI fix.)
- **(b) Graph reuse across the 10 denoise steps**: `llm_graph_input_state/action/sinusoidal_embedding/attn_no_cache_ae::can_reuse` are made to return `true` under proper guards (`llama-graph.cpp:109,137,192`; `llama-graph.h:100-105`), so the decoder graph is built once and replayed 9×. (Jetson-PI fix. This is the highest-value, lowest-risk optimization and should be measured FIRST, before any stage work.)
- **(c) Cross-tick double-buffered encoder KV**: tick N+1's VIT+encoder runs concurrently with tick N's K diffusion on separate streams. (Provider-internal pipeline.)
- **(d) Profiling shows encoder_time is a non-trivial hideable fraction of K×step_time** (i.e. the encoder is actually hideable rather than negligible). (Backfilled by §8.)
- **(e) Phase 6 memory-domain contract exists** so provider-owned ports can carry SWAP windows, OR new append-only ABI concepts (intermediate port direction, per-step `get_output`, stage-ready flag) are added — otherwise finer stages can only ever be provider-internal sub-steps hidden behind one `infer` stage, which is the current shape.

Even when (a)–(e) all hold, the correct implementation is a **provider-internal pipeline exposed to FlashRT as ONE callback stage** — not a FlashRT stage decomposition. The revisit question is therefore not "split into FlashRT stages" but "build the provider-internal cross-tick pipeline and re-measure."

**Recommended next actions (in priority order, all inside Jetson-PI, none a FlashRT stage split):**
1. Enable graph reuse across the 10 denoise steps (fix `can_reuse`).
2. Eliminate the cross-KV host round-trip (`llama-context.cpp:1069` D2H / `llama-graph.cpp:455` H2D feed-back) or at minimum measure it.
3. Move the per-step Euler recurrence on-device.
4. Collect the per-phase/per-step profile (§8) and re-evaluate this gate only if, after 1–3, the encoder (B) or a denoise step (C-inner) still dominates AND a cross-stage pipeline could hide it.

---

## 8. Profiling plan (for the main process to execute and backfill as profiling evidence)

The main process should add high-resolution `std::chrono` timestamp pairs (or `ggml_time_us()`) at the sites below, run ≥5 LIBERO ticks on the target Jetson with the Pi0-2.8B-F16 GGUF, and report per-tick and per-step wall times. All sites are in the Jetson-PI repo unless noted; the FlashRT-side sites confirm the stage boundary is negligible today.

### 8.1 Per-phase wall time (the critical-path attribution)

| Site | file:line | What to timestamp |
|---|---|---|
| Phase A start / end | `Jetson-PI/tools/mtmd/mtmd-helper.cpp:574` (`t_vit_start`) / `:576` (`t_vit_end`) | Per-view VIT. Already prints "Vit took" at `:579` — aggregate per tick (sum over views). |
| Phase B start / end | `Jetson-PI/tools/mtmd/mtmd-helper.cpp:765` (`t_encode_start`) / `:771` (`t_encode_end`) | Encoder (llm_build_pi0). Already prints "Encode took" at `:781`. |
| Phase C (whole loop) start / end | `Jetson-PI/tools/mtmd/mtmd-helper.cpp:774` (`t_decode_start`) / `:778` (`t_decode_end`) | All 10 denoise steps. Already prints "Decode took" at `:782`. |
| Phase C end (authoritative) | `Jetson-PI/src/llama-context.cpp:1388-1389` (`action_ready=true`) | Pair with loop start at `:1241` for total diffusion time independent of the mtmd wrapper. |
| Phase E | `Jetson-PI/src/jetson_pi_pi0.cpp:328` (call) / `:342` (return) | Confirm readout is negligible (pure memcpy at `llama-context.cpp:804`). |

### 8.2 Per-denoise-step wall time (THE critical measurement)

| Site | file:line | What to timestamp |
|---|---|---|
| Step start | `Jetson-PI/src/llama-context.cpp:1243` (`step_decode_start`) | BEGIN of denoise step i. |
| Step end | `Jetson-PI/src/llama-context.cpp:1268` (`step_decode_end`) | END of denoise step i. |
| Step duration | `Jetson-PI/src/llama-context.cpp:1270` (`step_decode_ms`) | **Currently computed but NOT printed.** ADD a `printf("step %d decode: %.3f ms\n", i, step_decode_ms);` here. This is the single most important missing print. |

**Measurement that confirms the critical-path hypothesis:** collect 10 `step_decode_ms` samples per tick across ≥5 ticks. Expectation: roughly constant across steps. If `sum(step_decode_ms) ≈ Decode took`, the diffusion loop is confirmed as the critical path and the per-step cost is attributable.

### 8.3 Graph-build vs graph-compute split (the reuse-fix justification)

Inside `process_ubatch` (`Jetson-PI/src/llama-context.cpp`), the rebuild branch is `:864-887`. To separate graph-build from graph-compute:

| Site | file:line | What to timestamp |
|---|---|---|
| After `model.build_graph` | `Jetson-PI/src/llama-context.cpp:872` (after `gf = model.build_graph(gparams)`) | Graph construction time. |
| After `ggml_backend_sched_alloc_graph` | `Jetson-PI/src/llama-context.cpp:877` (after alloc) | Sched-alloc time. |
| After `graph_compute` | `Jetson-PI/src/llama-context.cpp:898` (after `graph_compute`) | Pure compute time. |
| Reuse branch taken? | `Jetson-PI/src/llama-context.cpp:860` (`can_reuse` check) | Log whether the reuse or rebuild branch fired per step. Expectation today: rebuild 10× (because `can_reuse` returns false for state/action/sinusoidal/attn_no_cache_ae). |

**Measurement that confirms the reuse-fix lever:** if `(build_graph + sched_alloc)` at `:872-877` is a non-trivial fraction of per-step wall, the graph-reuse fix (making the four `can_reuse` paths return `true` under guards) is the real win and should be tried before any stage work. This refutes stage-splitting as the lever and routes the optimization to Jetson-PI.

### 8.4 The per-step D2H serialization barrier (the cross-tick-overlap blocker)

| Site | file:line | What to timestamp |
|---|---|---|
| D2H of `res->action` (velocity) | `Jetson-PI/src/llama-context.cpp:1254-1260` (`ggml_backend_tensor_get_async`) | Per-step device→host copy of `[action_dim*(action_steps+1)]` = `[32*51]` F32. Measure host-side wall (the `_get_async` name is misleading — it is consumed immediately by the Euler update, so it is effectively synchronous). |
| Host Euler update | `Jetson-PI/src/llama-context.cpp:1263-1264` (`cross.action[j] -= ...`) | Host recurrence. Usually cheap but verify it is not the bottleneck between steps. |
| H2D feed-back of `cross.action` next step | `Jetson-PI/src/llama-graph.cpp:129-134` (`llm_graph_input_action::set_input`, `ggml_backend_tensor_set`) | Per-step host→device re-upload of the latent `[50,32]`. |

**Measurement that confirms the sync barrier:** if `D2H(:1254) + H2D(:129) + host Euler(:1263)` per step is a measurable fraction of `step_decode_ms`, on-device recurrence is justified and is a prerequisite for any cross-tick overlap.

### 8.5 The cross-KV host round-trip (the B→C buffer-contract cost)

| Site | file:line | What to timestamp |
|---|---|---|
| Encoder GPU sync before KV extract | `Jetson-PI/src/llama-context.cpp:1057` (`synchronize()` inside encode, Pi0 branch) | Boundary between encoder GPU compute and host-side KV extraction. |
| Per-layer cross-KV D2H | `Jetson-PI/src/llama-context.cpp:1066-1071` (loop; `ggml_backend_tensor_get_async` at `:1069`) | Aggregate over n_layer=27 layers. Log total bytes (`ggml_nbytes(t_encoded_kv[i])` summed) and total ms. This is the B→C contract materialization cost. |
| H2D feed-back per denoise step | `Jetson-PI/src/llama-graph.cpp:455` (`ggml_backend_tensor_set` in `llm_graph_input_cross_kv_pi0::set_input`) | Per-step host→device feed of cross-KV. Note: cross-KV is constant across the 10 steps, so this is repeated work that graph reuse could eliminate. |

**Measurement that confirms the round-trip cost:** aggregate D2H bytes at `:1069` × 27 layers + aggregate H2D at `:455` × 10 steps. If non-trivial, eliminating the round-trip (keep cross-KV device-resident) is a Jetson-PI fix and a prerequisite for any zero-copy stage design.

### 8.6 FlashRT-side stage-boundary overhead (confirm splitting only adds overhead)

| Site | file:line | What to timestamp |
|---|---|---|
| Stage dispatch | `FlashRT/cpp/providers/llama_cpp/src/pi0_runtime.cpp:66` (`owner->engine.run_infer(...)`) | Whole-stage wall. |
| Monolithic infer | `FlashRT/cpp/providers/llama_cpp/src/jetson_pi_engine.cpp:242-247` (`jetson_pi_pi0_infer`) | The block a split would break. |
| Output readback | `FlashRT/cpp/providers/llama_cpp/src/jetson_pi_engine.cpp:264-290` (memcpy at `:289`) | Provider→host output copy. |
| Input staging (image swizzle) | `FlashRT/cpp/providers/llama_cpp/src/jetson_pi_engine.cpp:170-193` (`view_to_rgb` loop) | Host-side pixel swizzle a finer split would isolate into `encode_images` staging. |

**Measurement that confirms the stage boundary is negligible today:** the gap between `pi0_runtime.cpp:66` dispatch and `jetson_pi_engine.cpp:242` infer should be sub-millisecond, i.e. splitting adds boundary overhead with no offsetting overlap.

### 8.7 Decision-relevant aggregations to report

1. `encoder_time / (K × step_time)` ratio — if `encoder_time << K×step_time`, cross-tick pipelining has bounded value and the only theoretical stage-split win is small. If `encoder_time` is a meaningful fraction, revisit condition (d) is plausible.
2. `graph_build_time / step_time` per step — if non-trivial, the reuse fix (§7 action 2) is the lever, not a stage split.
3. `cross_kv_D2H + cross_kv_H2D_per_step` — sizes the B→C round-trip; if non-trivial, on-device cross-KV is a prerequisite.
4. `per_step_D2H_action + host_Euler + H2D_action` per step — sizes the sync barrier; if non-trivial, on-device recurrence is a prerequisite.
5. `sum(step_decode_ms) vs Decode took` — confirms the loop is the critical path and steps are equal-cost.

**Hypothesis to confirm or refute:** "Under current conditions, the diffusion K-loop dominates the tick, the dominant intra-loop inefficiency is graph rebuild + per-step D2H (both fixable inside Jetson-PI without a FlashRT stage split), and the encoder is not hideable in a useful intra-tick way." If the profile confirms this, the no-go stands. If it surprisingly shows the encoder is a large hideable fraction AND on-device recurrence + double-buffered KV are in place, only then revisit — and even then as a provider-internal pipeline behind one `infer` stage.


---

## 9. Phase 5 decision (recorded 2026-07-07)

**Decision: DEFER Phase 5 (no-go under current conditions).**

The four-gate test fails on scheduling benefit, buffer contract, and sync
semantics; profiling evidence is not yet collected on the Jetson target (the
structural case is strong enough to decide now). Phase 5 is therefore closed
as an evaluation/decision deliverable — no FlashRT stage split is implemented.

### Routing of the real optimization work

The latency wins the analysis identified all live **inside Jetson-PI**, behind
the existing single `infer` callback stage — they are NOT FlashRT stage
decompositions. Priority order (Jetson-PI-internal, future work, not part of
this migration phase):

1. **Graph reuse across the 10 denoise steps** — make
   `llm_graph_input_state`/`_action`/`_sinusoidal_embedding`/
   `_attn_no_cache_ae::can_reuse` return `true` under guards
   (`Jetson-PI/src/llama-graph.cpp:109,137,192`; `llama-graph.h:100-105`) so
   the decoder graph builds once and replays 9x. Highest value, lowest risk.
2. **On-device recurrence** — eliminate the per-step D2H of `res->action` +
   host Euler update (`llama-context.cpp:1254,1263-1264`) so `cross.action`
   stays GPU-resident. Prerequisite for any cross-tick overlap.
3. **Cross-KV round-trip** — eliminate or measure the device->host->device
   cross-KV hop (`llama-context.cpp:1069` D2H / `llama-graph.cpp:455` H2D per
   step; the `// TODO: tmp` hack at `llama-graph.h:56-60`).
4. **Cross-tick pipelining (only after 1-3)** — double-buffered encoder KV so
   tick N+1 VIT+encoder overlaps tick N diffusion on separate streams. Even
   then, expose to FlashRT as ONE callback stage.

### Conditions for revisiting Phase 5

Revisit only if ALL hold (from section 7): (a) on-device recurrence, (b) graph
reuse across denoise steps, (c) cross-tick double-buffered encoder KV, (d)
profiling shows encoder_time is a non-trivial hideable fraction of
K*step_time, (e) Phase 6 memory-domain contract exists (or new append-only
ABI: intermediate port direction, per-step `get_output`, stage-ready flag).
Even then the correct implementation is a provider-internal pipeline behind one
`infer` stage — not a FlashRT stage decomposition.

### Baseline sanity check (CPU, this host, 2026-07-07)

A baseline Pi0 run on this host (CPU, pi0_base F16 + pi0_base mmproj, LIBERO
fixture, action_steps=10/action_dim=32) confirmed the provider runs end-to-end
and surfaced the coarse phase timings the code already prints. The two-view VIT
(Phase A) measured ~6.8s and ~6.7s per view on CPU — dominated by the lack of
GPU and by per-step graph rebuild, consistent with the structural finding that
the real levers are Jetson-PI-internal (graph reuse, on-device recurrence), not
a FlashRT stage split. The full per-step profile (section 8) is to be collected
on the Jetson target where the deployment runs; on this CPU host the absolute
numbers are not deployment-representative, so they are not recorded here as the
profiling-evidence gate.

### FlashRT ABI gaps that would block finer stages (Phase 6 territory)

- No intermediate (bidirectional) port direction: frt_rt_port_direction is a strict 2-value enum (FRT_RT_PORT_IN=0 / FRT_RT_PORT_OUT=1), not a bitmask (FlashRT/runtime/include/flashrt/model_runtime.h:101). A port cannot be stage-N-out AND stage-(N+1)-in, so inter-stage buffers (image embeddings, encoder KV, diffusion latent) cannot be expressed as observable ports.
- No partial / per-step / streaming output: get_output(self, port, out, capacity, written, stream) is a full-snapshot read of one port by index (FlashRT/runtime/include/flashrt/model_runtime.h:201-203/222-224). No step-k/K parameter, no per-step ready flag, no streaming token. Per-denoise-step or partial action-chunk progress cannot be surfaced.
- Provider-owned path forbids inter-stage buffers and zero-copy windows: the builder rejects all streams/graphs/buffers/regions on provider-owned runtimes (FlashRT/runtime/src/model_runtime.cpp:203-208) and rejects any SWAP window / offset / bytes on ports (model_runtime.cpp:209-217). Hot intermediate tensors (diffusion noise, encoder KV) cannot get a fast lane. The docs explicitly defer this to Phase 6 (docs/model_runtime_api.md:127-129).
- No stage-ready / stage-aware invalidation signal: the only readiness signal is the whole-actions size check (jetson_pi_engine.cpp:285); invalidation is the blunt actions_buf.clear() on any set_input (jetson_pi_engine.cpp:157). A multi-stage split needs per-stage readiness flags and stage-aware invalidation, neither present.
- Engine vtable is monolithic: frt_llama_cpp_engine_v1 exposes only set_input / run_infer / get_output (FlashRT/cpp/providers/llama_cpp/include/flashrt/providers/llama_cpp/c_api.h:96-115); run_infer is a single opaque verb. Backing N stages needs either a generic run_stage(stage_idx) engine verb or per-sub-stage verbs, and Jetson-PI must expose sub-phase C entry points (jetson_pi_pi0_encode_images / _prefill_context / _diffusion_step / _decode_actions) to replace the single jetson_pi_pi0_infer (jetson_pi_engine.cpp:242-247).
- No shape-bucket prepare for sub-stages: provider returns -3 unsupported_prepare (pi0_runtime.cpp:19-23). If a split introduced variable sub-stage shapes (e.g. per-step), there is no capture-on-miss path.
