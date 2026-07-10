# Phase 8 Evaluation — Zero-copy / DLPack / host SWAP for provider-owned ports (TODO-5)

> Status: 2026-07-08. Evaluates CLAUDE.md **TODO-5** (§11 DLPack / host SWAP zero-copy). Mirrors the Phase 5 / Phase 7 (TODO-2) evaluation structure.

## 1. Executive summary + recommendation

**Recommendation: NO-GO. Defer TODO-5 under current conditions.**

The Phase 6 memory-domain contract (commit `5d25672`/`b9919be`) plus its real token consumer (commit `95d7175`, Pi0 actions OUT port, `HOST_VISIBLE`) already cover the honest zero-copy-adjacent case: a host can read the actions buffer without a device sync, via `copy_to_host`. No FlashRT caller reads the actions buffer via DLPack, via a raw SWAP window, or as a device-resident tensor today. TODO-5's own prerequisite in CLAUDE.md — "明确真实 caller 需求（谁要零拷贝读 actions？当前 consumer 都是 copy_to_host）" — answers itself: there is no such caller.

Adding DLPack export or a host SWAP window now would be to add a **second, half-tested buffer form for a nonexistent consumer**, which directly violates the contract's founding principle ("不要在没有 memory-domain contract 前假装可以零拷贝共享 buffer" — and once you HAVE the contract, don't add a parallel half-finished one either). The four lenses do not converge on GO: only sync semantics (DLPack is a well-defined capsule) passes; scheduling benefit, buffer-contract-need, and ABI cost all fail under current callers. This is the same gate Phase 5 (Pi0 stage split) and Phase 7 (LLM repeatable decode) hit: *implementable ≠ justified*.

A DEFER here routes the work to **when a real caller needs it** (revisit conditions a–c, §6) and keeps the provider surface honest: one buffer form (`HOST_VISIBLE` token + `copy_to_host`), exercised and parity-verified, not two.

## 2. Gate finding (verified against source)

**The only token consumer today** is the Pi0 actions OUT port (commit `95d7175`, `cpp/providers/llama_cpp/src/jetson_pi_engine.cpp`):
- `pi0_token_copy_to_host` (:307) — memcpy from the live `actions_buf` into a caller host buffer.
- `pi0_token_copy_from_host` (:321) — returns `-3` unsupported (OUT port).
- `pi0_token_sync` (:327) — `return 0` (HOST_VISIBLE no-op).
- `destroy` = nullptr.

`location_kind = FRT_RT_LOCATION_HOST_VISIBLE` (`model_runtime.h:120`: "copy_to_host is a plain read; no sync"). The backing store is a host `std::vector<float>`. **Zero callers** use a DLPack capsule, a raw `frt_buffer` SWAP window, or a `DEVICE_LOCAL` token against this port — verified by `grep -rnE 'dlpack|DLPack|swap_window|cap_swap'` across `cpp/providers` + `flash_rt/frontends/jetson_pi` returning nothing relevant (the only `from_dlpack` hits are in `csrc/attention/flash_attn_4_src/flashrt_fa4/cute/` — CUTLASS/CuTe internals for FA4 kernels, unrelated to the Jetson-PI provider).

**Provider-owned ports are forced STAGED** (`model_runtime.cpp:184` `d.buffer = nullptr` in `frt_runtime_builder_add_port_token`): there is no raw `frt_buffer` SWAP window on a token port by construction. The `HOST_VISIBLE` token already advertises the honest capability ("host can read without a device sync") that every actual consumer uses.

## 3. Current surface

- **Phase 6 contract**: opaque `frt_memory_token` + `frt_memory_token_verbs` (`copy_to_host`/`copy_from_host`/`sync`/`destroy`) + `frt_rt_location_kind` (`HOST_VISIBLE`/`DEVICE_LOCAL`), parallel `port_tokens` array on `frt_model_runtime_v2` (`model_runtime.h:183-229`, `:378`).
- **Real consumer**: Pi0 actions port mints a `HOST_VISIBLE` token (commit `95d7175`); the C++ e2e test sub-test E reads it via `copy_to_host` and asserts byte-identity vs `get_output`.
- **No DLPack path** exists for provider-owned ports; no `__dlpack_managed`/`from_dlpack` bridge in the Python frontend (`flash_rt/frontends/jetson_pi/pi0.py` returns a NumPy array copied out of `get_output`).

§11 gap: §11 says "后续如果需要 CPU SWAP 或零拷贝，FlashRT buffer descriptor 需要增加 memory domain" and "在没有 memory-domain 契约前，不应将普通 CPU 指针伪装成 FlashRT device buffer". The memory-domain contract now EXISTS (Phase 6); the question is whether to **use** it for zero-copy yet. The answer is no, because no consumer asks.

## 4. Four-lens evaluation

### Lens 1 — Scheduling / performance benefit
None under current callers. The `copy_to_host` of a `[10, 32]` F32 action chunk (1.28 KB) is negligible vs the ~1.1 s Pi0 tick; a zero-copy read would save a 1.28 KB memcpy — unmeasurable. A device-resident read would matter only if a downstream consumer runs on the same GPU and currently pays a D2H it could avoid — but no such consumer exists. **Lens 1: FAILS.**

### Lens 2 — Buffer contract
Half-met. The contract is in place and COULD carry a DLPack capsule (a new verb `to_dlpack` minting a managed tensor, or a `DEVICE_LOCAL` token for a CUDA-resident buffer). But adding it now creates a second buffer form alongside the exercised `HOST_VISIBLE`+`copy_to_host` path — a half-tested parallel surface for a nonexistent consumer. The contract's own principle ("don't pretend zero-copy; route copies through explicit verbs with explicit location_kind") argues against adding a SWAP window until a consumer needs it. **Lens 2: half-met → don't add a half-finished SWAP.**

### Lens 3 — Sync semantics (the lens that PASSES)
DLPack is a well-defined, language-agnostic capsule (`DLManagedTensor` with `deleter`); `HOST_VISIBLE` already means "no device sync needed"; `DEVICE_LOCAL` would require a `sync` before host read (the contract already models this). The semantics are clean and implementable. **Lens 3: PASSES** — TODO-5 is *implementable*, not *incoherent*. (Same position as Phase 7's lens 3.)

### Lens 4 — ABI cost
Real. A DLPack export on a provider-owned port is new ABI surface: a `to_dlpack`/`from_dlpack` verb pair (or a new `location_kind` + capsule-manage verbs), dtype/shape/strides negotiation, and a managed-lifetime contract (who calls `deleter`, how it interlocks with the holder refcount). For zero consumers, this is uncompensated ABI surface that future append-only extensions must stay compatible with. **Lens 4: real cost, no consumer to justify it.**

## 5. Recommendation + revisit conditions

**DEFER.** The four lenses do not converge on GO: lens 1 (benefit) fails, lens 2 (contract) is half-met and the contract's own principle argues against a premature SWAP, lens 4 (ABI) is real. Only lens 3 (sync semantics) passes — making TODO-5 *implementable* but not *justified*, exactly the Phase 5 / Phase 7 pattern.

**Revisit (GO when ANY of):**
- **a.** A real caller needs NumPy/PyTorch **zero-copy of the actions buffer** — e.g. a downstream policy controller running at kHz that cannot afford the (currently 1.28 KB) `copy_to_host` memcpy, or a learning loop that materializes the action as a trainable tensor without copy. Lens 1 → GO.
- **b.** A caller needs a **device-resident actions read** — a second provider (or a FlashRT CUDA-Graph stage) consuming Pi0's actions as its input on the same GPU, avoiding a D2H+H2D round-trip. Then a `DEVICE_LOCAL` token (with a real `sync` verb) is justified; lens 2 → GO.
- **c.** **Cross-provider tensor hand-off via DLPack** becomes a FlashRT requirement — two providers exchanging a tensor through a DLPack capsule rather than a host staging copy. Lens 2/4 → GO (the ABI cost is paid by a real interop need).

Until one of a–c, the current `HOST_VISIBLE` token + `copy_to_host` (byte-parity-verified by sub-test E and TODO-1 parity) is the honest, sufficient zero-copy-adjacent face.

## 6. Note on the Phase 5 / Phase 7 precedent

This mirrors Phase 5 (commit `8a4cb9d`) and Phase 7/TODO-2 (commit `8dfb52d`): a phase resolved by **evaluation/decision**, not implementation, with explicit revisit conditions. The shared gate is "no current caller benefits / no current consumer needs the surface". The difference: Phase 5's sync-semantics lens FAILED (Pi0 flow-matching has no usable intermediate state); Phase 7's and Phase 8's lens 3 PASS (LLM decode / DLPack capsules are clean boundaries) — so TODO-5 is *implementable* where Phase 5 was *incoherent*, but the DEFER conclusion holds on the same "no caller need" ground. The revisit is gated on a **caller need** (a–c), like Phase 7, not on a profiling signal (Phase 5's was).
