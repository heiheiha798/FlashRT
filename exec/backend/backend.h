/* FlashRT exec — internal hardware backend interface.
 *
 * The whole hardware surface the execution contract needs, as a small set of
 * free functions. CUDA-free here (all handles are void*) so the core C ABI in
 * src/ never includes a vendor runtime header. One backend translation unit
 * (backend/cuda/cuda_backend.cpp today; backend/hip/... later) implements it.
 *
 * All functions return null / false on failure; the core layer turns that into
 * the public frt_status codes. No exceptions cross this boundary.
 *
 * Phase 6 clarification — the surface splits into two groups:
 *
 *   (a) SHARED primitives (every backend implements): malloc/free, stream_*,
 *       event_*, memcpy_dtod_async, memset_async. A provider-owned runtime
 *       (e.g. Jetson-PI/GGML) does NOT consume these — GGML owns its device
 *       memory and streams internally — so FlashRT never calls this group for
 *       the provider-owned path. It is the CUDA-export path's vocabulary only.
 *
 *   (b) CUDA-GRAPH-CAPTURE primitives (optional): capture_begin/end,
 *       graph_exec_destroy, graph_launch. A backend whose capture_begin
 *       returns false cannot host graph-capture stages; it must use the
 *       provider-owned callback-stage path (frt_runtime_stage_desc_v2 with
 *       kind=CALLBACK) instead. GGML has no cudaGraph_t analogue, so a GGML-
 *       facing exec backend would implement group (a) only and never (b) —
 *       but no such backend is needed in this migration because GGML is
 *       integrated as a model provider (not a bare exec backend). See
 *       docs/phase6_backend_vtable_eval.md.
 *
 * Backend selection is LINK-TIME: one TU per build (exec/CMakeLists.txt links
 * cuda/cuda_backend.cpp). Runtime multi-backend selection in one process would
 * require promoting this namespace to a registered function-pointer table; that
 * is out of scope for Phase 6 (no consumer — see the eval). The import() seam
 * below is declared so a future cross-domain buffer registration path has a
 * named extension point, but it is NOT implemented for non-CUDA backends in
 * this phase.
 */
#ifndef FLASHRT_EXEC_BACKEND_H
#define FLASHRT_EXEC_BACKEND_H

#include <stddef.h>

namespace frt {
namespace be {

/* ── (a) SHARED primitives — every backend implements ─────────────────── */

/* --- device memory --- */
void* malloc(size_t bytes);          /* device alloc; null on failure        */
void  free(void* dptr);

/* --- streams (handles owned by ctx) --- */
void* stream_create(int priority);   /* prioritized stream; null on failure   */
void  stream_destroy(void* stream);
void  stream_sync(void* stream);

/* --- events --- */
void* event_create();
void  event_destroy(void* event);
bool  event_record(void* event, void* stream);
bool  stream_wait_event(void* stream, void* event);

/* --- async copies / fills (allocation-free; capture-safe) --- */
bool  memcpy_dtod_async(void* dst, const void* src, size_t bytes, void* stream);
bool  memset_async(void* dptr, int value, size_t bytes, void* stream);

/* ── (b) CUDA-GRAPH-CAPTURE primitives — optional ───────────────────────
 * capture_begin/end wrap stream-level capture (RELAXED mode); capture_end
 * instantiates and returns an executable graph handle, freeing the transient
 * non-executable graph. graph_exec_destroy frees the executable. A backend
 * that cannot capture returns false from capture_begin; the caller then uses
 * the provider-owned callback-stage path instead (no graph object involved). */
bool  capture_begin(void* stream);
void* capture_end(void* stream);     /* returns graph-exec handle; null = fail */
void  graph_exec_destroy(void* graph_exec);
bool  graph_launch(void* graph_exec, void* stream);

/* ── future extension point (DECLARED ONLY, Phase 6) ────────────────────
 * Register an externally-owned memory region (e.g. a provider's device buffer)
 * so a CUDA-export graph could bind it as a SWAP window without a copy. NOT
 * implemented for non-CUDA backends in this phase — cross-backend zero-copy
 * needs a memory-domain contract that FlashRT only just gained at the
 * provider-owned port layer (frt_memory_token). This declaration exists so a
 * later phase can add the implementation without retroactive header churn. */
/* void* import(void* external_handle, size_t bytes, uint32_t location_kind); */

}  // namespace be
}  // namespace frt

#endif  /* FLASHRT_EXEC_BACKEND_H */
