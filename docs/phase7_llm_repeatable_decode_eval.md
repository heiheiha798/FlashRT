# Phase 7 Evaluation — LLM Repeatable Decode (prefill / decode-step / logits)

> Status: 2026-07-07. Evaluates CLAUDE.md **TODO-2** (§8.1 LLM `prefill once → decode (host-repeatable)` + `next_token`/`logits` ports). Mirrors the Phase 5 (finer-Pi0-stages) evaluation structure.

## 1. Executive summary + recommendation

**Recommendation: NO-GO. Defer TODO-2 under current conditions.**

Four lenses (scheduling benefit, buffer contract, sync semantics, ABI cost) do **not** converge on GO. The split is *implementable* (lens 3 passes cleanly — unlike Phase 5, where Pi0's flow-matching has no usable intermediate state, an LLM's per-token decode IS the natural boundary), but it is not *justified under current callers*: no FlashRT caller multiplexes LLM sessions or interleaves LLM decode with another provider, so the "host interrupts/schedules at token boundaries" benefit (the entire point of §8.1's repeatable decode) is theoretical. Meanwhile the cost is real: a Jetson-PI narrow-API extension (`prefill`/`decode_step`/`reset`), an append-only FlashRT engine-vtable ABI extension (`run_infer` → `run_stage`), new ports/stages, and a refactor of the one-shot `generate` path that TODO-1 (commit `0cee8c7`) just proved byte-identical against the native API.

The single most important structural fact: the Jetson-PI narrow C API `jetson_pi_llm` exposes only `open/generate/close/open_error` (`Jetson-PI/include/jetson_pi_llm.h`). `generate` hardcodes `llama_memory_clear(..., true)` (`Jetson-PI/src/jetson_pi_llm.cpp:200`) and `llama_sampler_reset` (`:201`) every call — KV and sampler state are wiped per completion. There is **no** `prefill`/`decode_step`/`reset`/`get_logits` symbol. FlashRT cannot fabricate the prefill/decode boundary from its side; the only seam is `generate`, and everything inside it is opaque. This is the same gate Phase 5 hit: a stage split requires Jetson-PI to expose separate sub-phase entry points.

A DEFER here is the correct, defensible outcome — it prevents adding ABI surface, new ports, and a 2-repo refactor for a benefit no caller exercises, exactly as Phase 5's DEFER prevented Pi0 stage-splitting for zero latency gain. It routes the work to **when a real caller needs it** (revisit conditions a–d, §6).

## 2. Gate finding (verified against source)

**Jetson-PI narrow API surface** (`Jetson-PI/include/jetson_pi_llm.h`, 98 lines): `jetson_pi_llm_open` (h:55), `close` (h:59), `last_error` (h:64), `open_error` (h:69), `generate` (h:89). Header doc (h:11-13, h:88) is explicit: "Each generate call clears KV state and runs an independent first-turn completion (no multi-turn state)." Config (`jetson_pi_llm_config`, h:28-42) carries `temp/top_k/top_p/seed/max_tokens` — no flags for logits exposure, sampler rebuild, or step mode.

**`generate` internals** (`Jetson-PI/src/jetson_pi_llm.cpp:174-277`):
- KV wipe: `llama_memory_clear(llama_get_memory(e->lctx), true)` at **:200**.
- Sampler wipe: `llama_sampler_reset(e->smpl)` at **:201**.
- Prefill: `llama_batch_get_one(prompt_tokens)` + `llama_decode` at **:220-225** (whole-prompt single batch).
- Decode loop (**:231-266**): `llama_sampler_sample` (:232) → EOG check via `llama_vocab_is_eog` (:233, break internally — host never sees it) → `llama_token_to_piece` detokenize (:242-256) → `llama_sampler_accept(new_token_id)` (:258, feeds token back into sampler state) → `llama_batch_get_one(&new_token_id,1)` + `llama_decode` (:261-265, single-token step).
- **Logits are never exposed**: zero `llama_get_logits`/`llama_get_logits_ith` calls in `jetson_pi_llm.cpp`. The host gets only the final detokenized text blob.

The llama.cpp primitives needed for a split are **all already linked**; all except `llama_get_logits_ith` are already used piecewise inside `generate` (`llama_decode` :222/:262, `llama_sampler_sample` :232, `llama_sampler_accept` :258, `llama_memory_clear` :200, `llama_vocab_is_eog` :233). `llama_get_logits_ith` is available (declared in `llama-context.h`) but currently uncalled — the work is *exposing* these as host-driven entry points, not inventing capability.

## 3. Current FlashRT LLM provider surface

- **C ABI** (`cpp/providers/llama_cpp/include/flashrt/providers/llama_cpp/c_api.h`): LLM ports `FRT_LLAMA_CPP_LLM_PORT_PROMPT=0`, `PORT_TEXT=1` (h:29-32) — **no** `tokens` in, `next_token` out, or `logits` out. Stage `FRT_LLAMA_CPP_LLM_STAGE_INDEX_INFER=0` only (h:34-36) — **no** prefill/decode split. Engine vtable `frt_llama_cpp_engine_v1` (h:96-115) has `run_infer(void*)` only — **no stage parameter**.
- **Runtime** (`cpp/providers/llama_cpp/src/llm_runtime.cpp`): two STAGED ports (prompt in, text out, :131-138), one `infer` callback stage (:139-140), `run_stage` (:64-79) collapses every stage to `engine.run_infer`.
- **Engine** (`cpp/providers/llama_cpp/src/jetson_pi_engine.cpp`, LLM section :346-462): `LlmEngine` (:354-367) holds `jetson_pi_llm*`, prompt stash, `text_buf`. `llm_engine_run_infer` (:401-426) calls `jetson_pi_llm_generate` **once** (:415-417). Each `run_infer` is a full independent completion.
- **Python** (`flash_rt/frontends/jetson_pi/llm.py`): `generate(prompt)` (:103-127) = `set_input(PROMPT)` → `run_stage(INFER)` → `get_output(TEXT)`. Single whole-prompt completion.

§8.1 gap: §8.1 specifies `prompt`/`tokens`(opt in)/`next_token`(out)/`logits`(opt out) ports + `prefill once → decode (repeatable)` stages. Current has `prompt`+`text`+single `infer`. Every one of the five §8.1 items (tokens-in, next_token, logits, prefill/decode split, repeatable decode) is absent.

## 4. Four-lens evaluation

### Lens 1 — Scheduling benefit

§8.1's stated benefit (`jetsonpi迁移.txt:302`): "host 在 token 边界重复调用 decode，从而保留中断和调度能力". For a **single-stream completion** (one prompt → one text blob), there is no scheduling overlap to win: prefill and decode are strictly sequential within one session, and FlashRT has no second concurrent stream to pipeline against. The win only materializes when FlashRT actually schedules multiple LLM sessions or interleaves LLM decode with another provider's work — which no **jetson_pi** caller does. The one-shot `generate` already returns the full blob; a host wanting interrupt/resume would need FlashRT-level session multiplexing first. **Lens 1: FAILS under current callers.**

> Scope note: FlashRT's **torch** LLM frontends (`flash_rt/frontends/torch/qwen3_rtx.py` `set_prompt`/`reset_state`/`forward_own_decode_nvfp4`/`decode_step_with_graph` + n-gram speculative decode; `qwen36_rtx.py` MTP speculative decode `generate_own_speculative_K*` + committed-stream prefill/decode) already implement host-driven per-token decode — but on a different provider (`framework="torch"`, NVFP4 weights, not GGUF), constructed directly (not via `load_model(..., framework="jetson_pi")`). §8.1/TODO-2 is explicitly the **jetson_pi/llama.cpp** provider spec. The torch precedent shows the pattern is implementable and useful where there is demand; it does not create a jetson_pi caller need. A GGUF-LLM caller cannot use the torch path. This scoping is legitimate, not a dodge — but it must be stated, since the torch frontends are the obvious counterexample to a blanket "FlashRT has no per-step LLM decode" claim.

### Lens 2 — Buffer contract

A `next_token` (I32 out) port is trivial (4 bytes). `logits` (F32 out, vocab-sized) is the invasive piece: qwen3-0.6b vocab ~150k → ~600KB/step, with ownership/lifetime questions (per-step snapshot vs ring). The Phase 6 memory-domain token contract (commit `95d7175`) could carry a `logits` token, but no consumer needs it today. **Lens 2: contract defined (STAGED ports + token contract exist), but no consumer pulls logits.**

### Lens 3 — Sync semantics (the lens that PASSES, unlike Phase 5)

The boundary is clean: a `decode_step` would sample one token, accept it into the sampler, decode it, return `(token_id, eog_flag)`. KV and sampler state persist across steps (the whole point). EOG is observable (`llama_vocab_is_eog`). The semantics are well-defined and llama.cpp supports them directly. **This is where LLM differs from Pi0**: Pi0's flow-matching has no usable intermediate state (TTFA==TTC, Phase 5 §1); an LLM's per-token decode IS the natural boundary. **Lens 3: PASSES — TODO-2 is implementable, not incoherent.**

### Lens 4 — ABI cost

The engine vtable `frt_llama_cpp_engine_v1` (c_api.h:96-115) has only `run_infer(void*)`. To expose `prefill` + `decode` as two observable stages, the vtable needs an append-only `run_stage(void*, uint32_t stage)` verb (struct_size-gated, consistent with §10.1's append-only guidance, `jetsonpi迁移.txt:392-395`). Plus new port enums (`PORT_NEXT_TOKEN`, `PORT_LOGITS`) and stage enums (`STAGE_PREFILL`, `STAGE_DECODE`). The runtime wrapper `llm_runtime.cpp:64` already has a `run_stage` dispatcher that collapses every stage to `engine.run_infer` — so the vtable extension is the one FlashRT-ABI change. Append-only and safe, but it's a vtable shape change on a structure TODO-1's parity test and the fake-engine test both exercise. **Lens 4: real cost — one vtable ABI extension + 2-repo refactor of a path TODO-1 just stabilized.**

## 5. Cost / blast radius (if implemented — option iii MVP)

- **Jetson-PI** (`jetson_pi_llm.h/.cpp`): add `jetson_pi_llm_prefill`/`decode_step`/`reset` (+ optional `get_logits`); refactor `generate` into `prefill` + loop(`decode_step`) + `reset` (one code path). New Jetson-PI tests.
- **FlashRT C ABI** (`c_api.h`): `PORT_NEXT_TOKEN`/`PORT_LOGITS` enums + `STAGE_PREFILL`/`STAGE_DECODE` enums + engine-vtable `run_stage` verb (append-only, struct_size-gated).
- **FlashRT runtime** (`llm_runtime.cpp`): declare new ports + 2 stages; wire `run_stage` to the new engine verb.
- **FlashRT engine** (`jetson_pi_engine.cpp` LlmEngine): rework verbs for prefill/decode/`get_output(next_token, logits)`; stop forcing KV reset between calls.
- **Python** (`llm.py`): `prefill()`/`decode_step()`/`reset()`; `generate()` becomes a wrapper.
- **Tests**: streaming smoke + §14 KV-isolation gate (consecutive sessions must not leak — the §14 principle `jetsonpi迁移.txt:599` is stated for Pi0 context; the same isolation requirement applies to LLM KV across sessions). Existing `test_llama_cpp_jetson_pi_llm.cpp` (Phase 3, `083f6f8`) + `test_llama_cpp_jetson_pi_llm_parity.cpp` (TODO-1, `0cee8c7`) drive the old `STAGE_INFER`→`PORT_TEXT` path — must keep working (the `text` convenience face stays; if implemented, `generate()` becomes a wrapper over `prefill`+loop(`decode_step`)+`reset`).

Three repos touched; one vtable ABI extension; one refactor of a path TODO-1 just proved byte-identical. Non-trivial but bounded.

## 6. Recommendation + revisit conditions

**DEFER.** The four lenses do not converge on GO: lens 1 (scheduling benefit) fails under current callers; lens 2 (buffer contract) is half-met (no logits consumer); lens 4 (ABI cost) is real. Only lens 3 (sync semantics) passes — which makes TODO-2 *implementable* (not incoherent like a naive Pi0 split would be) but not *justified*.

**Revisit (GO when ANY holds):**
- **a.** A real FlashRT caller needs host-driven token looping — e.g. an LLM serving path with early-stop / speculative decode / multi-session KV reuse. Lens 1 → GO.
- **b.** FlashRT scheduler gains multi-session LLM multiplexing — prefill/decode becomes the natural scheduling unit. Lens 1 → GO. *(Caveat: as of 2026-07-07 no jetson_pi multi-session scheduler exists in FlashRT — grep for scheduler/multiplex/session in `flash_rt/`/`runtime/`/`cpp/` returns only diffusion samplers and unrelated comments; the only LLM serving path `examples/qwen3_openai_server.py` uses the torch frontend. So (b) is contingent on a jetson_pi serving path that does not exist today, and is near-redundant with (a) — if (a) fires, (b) is moot. It is a legitimate future condition, not an escape hatch.)*
- **c.** A caller needs logits (external sampler, classifier head, constrained decoding). Lens 2 → GO; the `logits` port + a `get_logits` Jetson-PI symbol become justified.
- **d.** LLM streaming UX becomes a requirement (token-by-token output to a downstream consumer that can't wait for the full blob). A product requirement, not a scheduling one, but a legitimate GO trigger.

Until one of a–d, the current one-shot `generate` (verified byte-parity by TODO-1, commit `0cee8c7`) is the honest, sufficient LLM face.

## 7. Note on the Phase 5 precedent

This mirrors Phase 5 (commit `8a4cb9d`, `docs/phase5_pi0_stages_eval.md`): a phase resolved by **evaluation/decision**, not implementation, with explicit revisit conditions. The Phase 5 gate required all four of (scheduling benefit, buffer contract, sync semantics, profiling evidence); here lens 3 passes where Phase 5's sync-semantics lens failed, but lens 1 fails where Phase 5's also failed — so the DEFER conclusion holds on the same "no current caller benefits" ground. The difference is that TODO-2's revisit is gated on a *caller need* (a–d), whereas Phase 5's is gated on a *profiling signal + Jetson-PI internal optimization* — because LLM decode is genuinely splittable (lens 3) in a way Pi0 diffusion is not.
