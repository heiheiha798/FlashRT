#ifndef FLASHRT_PROVIDERS_LLAMA_CPP_C_API_H
#define FLASHRT_PROVIDERS_LLAMA_CPP_C_API_H

#include "flashrt/model_runtime.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum frt_llama_cpp_model_family {
    FRT_LLAMA_CPP_MODEL_PI0 = 1,
};

enum frt_llama_cpp_pi0_port {
    FRT_LLAMA_CPP_PI0_PORT_IMAGES  = 0,
    FRT_LLAMA_CPP_PI0_PORT_PROMPT  = 1,
    FRT_LLAMA_CPP_PI0_PORT_STATE   = 2,
    FRT_LLAMA_CPP_PI0_PORT_ACTIONS = 3,
};

enum frt_llama_cpp_pi0_stage_index {
    FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER = 0,
};

typedef struct frt_llama_cpp_pi0_config {
    uint32_t struct_size;

    const char* model_path;
    const char* mmproj_path;
    const char* backend;

    uint32_t n_views;
    uint32_t image_height;
    uint32_t image_width;
    uint32_t image_channels;
    uint32_t state_dim;
    uint32_t action_steps;
    uint32_t action_dim;
} frt_llama_cpp_pi0_config;

typedef struct frt_llama_cpp_engine_v1 {
    uint32_t struct_size;
    uint32_t reserved;

    void* self;
    /* retain/release must be both null for a borrowed engine, or both set for
     * a reference-counted engine. Asymmetric ownership hooks are rejected. */
    void (*retain)(void* self);
    void (*release)(void* self);

    int (*set_input)(void* self, uint32_t port,
                     const void* data, uint64_t bytes, int stream);
    int (*run_infer)(void* self);
    int (*get_output)(void* self, uint32_t port,
                      void* out, uint64_t capacity, uint64_t* written,
                      int stream);
    /* Should return a non-null borrowed string. If it violates that contract,
     * the wrapper reports a stable boundary error string. */
    const char* (*last_error)(void* self);
} frt_llama_cpp_engine_v1;

typedef struct frt_llama_cpp_engine_factory_v1 {
    uint32_t struct_size;
    uint32_t reserved;

    void* self;
    /* Returns one owned engine reference in out_engine. The config pointer
     * and its string fields are borrowed and valid only for the duration of
     * the callback; factories must copy anything they keep. The runtime-open
     * wrapper consumes the returned reference after retaining the model
     * runtime's own engine reference. Factory-created engines must provide
     * symmetric retain/release hooks; borrowed engines are only accepted by
     * the lower create_with_engine entry point. On nonzero return, no engine
     * ownership is transferred and out_engine is ignored. */
    int (*create_pi0)(void* self, const frt_llama_cpp_pi0_config* config,
                      frt_llama_cpp_engine_v1* out_engine);
    const char* (*last_error)(void* self);
} frt_llama_cpp_engine_factory_v1;

int frt_llama_cpp_pi0_runtime_create_with_engine(
    const frt_llama_cpp_pi0_config* config,
    const frt_llama_cpp_engine_v1* engine,
    frt_model_runtime_v2** out);

/* Provider-specific JSON open path for the current dependency-injection
 * boundary. Required JSON fields:
 *   model_family="pi0", model_path, mmproj_path, backend,
 *   n_views, image_height, image_width, image_channels,
 *   state_dim, action_steps, action_dim.
 * No field has a default; missing or mismatched fields fail hard. */
int frt_llama_cpp_pi0_runtime_open_with_engine_factory(
    const char* config_json,
    const frt_llama_cpp_engine_factory_v1* factory,
    frt_model_runtime_v2** out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLASHRT_PROVIDERS_LLAMA_CPP_C_API_H */
