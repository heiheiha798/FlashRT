#include "flashrt/providers/llama_cpp/c_api.h"

#include <cstddef>
#include <cstring>
#include <new>
#include <string>

namespace {

struct RuntimeOwner {
    frt_llama_cpp_engine_v1 engine{};
    std::string last_error;
    int64_t image_shape[4] = {};
    int64_t state_shape[1] = {};
    int64_t action_shape[2] = {};
};

int unsupported_prepare(void* self, uint32_t, frt_shape_key) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (owner) owner->last_error = "llama_cpp Pi0 provider has no graph variants";
    return -3;
}

const char* engine_error(RuntimeOwner* owner) {
    const char* err = owner->engine.last_error(owner->engine.self);
    return err ? err : "llama_cpp Pi0 engine returned a null error";
}

int set_input(void* self, uint32_t port, const void* data, uint64_t bytes,
              int stream) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return -1;
    const int rc = owner->engine.set_input(owner->engine.self, port, data,
                                           bytes, stream);
    if (rc != 0) {
        owner->last_error = engine_error(owner);
    } else {
        owner->last_error.clear();
    }
    return rc;
}

int get_output(void* self, uint32_t port, void* out, uint64_t capacity,
               uint64_t* written, int stream) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return -1;
    const int rc = owner->engine.get_output(owner->engine.self, port, out,
                                            capacity, written, stream);
    if (rc != 0) {
        owner->last_error = engine_error(owner);
    } else {
        owner->last_error.clear();
    }
    return rc;
}

int run_stage(void* self, uint32_t stage, int stream) {
    (void)stream;
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return -1;
    if (stage != FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER) {
        owner->last_error = "unknown llama_cpp Pi0 stage";
        return -1;
    }
    const int rc = owner->engine.run_infer(owner->engine.self);
    if (rc != 0) {
        owner->last_error = engine_error(owner);
    } else {
        owner->last_error.clear();
    }
    return rc;
}

int step(void* self) {
    return run_stage(self, FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER, -1);
}

const char* last_error(void* self) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return "null llama_cpp Pi0 runtime";
    if (!owner->last_error.empty()) return owner->last_error.c_str();
    return engine_error(owner);
}

void destroy_owner(void* self) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (owner->engine.release) owner->engine.release(owner->engine.self);
    delete owner;
}

}  // namespace

extern "C" int frt_llama_cpp_pi0_runtime_create_with_engine(
        const frt_llama_cpp_pi0_config* config,
        const frt_llama_cpp_engine_v1* engine,
        frt_model_runtime_v2** out) {
    if (!out) return -1;
    *out = nullptr;
    if (!config || config->struct_size < sizeof(frt_llama_cpp_pi0_config) ||
        !config->model_path || !config->model_path[0] ||
        !config->mmproj_path || !config->mmproj_path[0] ||
        !config->backend || !config->backend[0] || !config->n_views ||
        !config->image_height || !config->image_width ||
        !config->image_channels || !config->state_dim ||
        !config->action_steps || !config->action_dim) {
        return -1;
    }
    if (!engine || engine->struct_size < sizeof(frt_llama_cpp_engine_v1) ||
        !engine->self || !engine->set_input || !engine->run_infer ||
        !engine->get_output || !engine->last_error ||
        static_cast<bool>(engine->retain) !=
            static_cast<bool>(engine->release)) {
        return -1;
    }

    auto* owner = new (std::nothrow) RuntimeOwner();
    if (!owner) return -5;
    owner->engine = *engine;
    if (owner->engine.retain) owner->engine.retain(owner->engine.self);

    owner->image_shape[0] = static_cast<int64_t>(config->n_views);
    owner->image_shape[1] = static_cast<int64_t>(config->image_height);
    owner->image_shape[2] = static_cast<int64_t>(config->image_width);
    owner->image_shape[3] = static_cast<int64_t>(config->image_channels);
    owner->state_shape[0] = static_cast<int64_t>(config->state_dim);
    owner->action_shape[0] = static_cast<int64_t>(config->action_steps);
    owner->action_shape[1] = static_cast<int64_t>(config->action_dim);

    frt_runtime_builder b = frt_runtime_builder_create_provider_owned();
    if (!b) {
        destroy_owner(owner);
        return -5;
    }

    int rc = 0;
    const int64_t prompt_shape[1] = {-1};
    rc |= frt_runtime_builder_add_port(
        b, "images", FRT_RT_MOD_IMAGE, FRT_RT_DTYPE_U8, FRT_RT_LAYOUT_NHWC,
        FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1, owner->image_shape, 4, 0,
        nullptr, 0, 0);
    rc |= frt_runtime_builder_add_port(
        b, "prompt", FRT_RT_MOD_TEXT, FRT_RT_DTYPE_U8, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1, prompt_shape, 1, 0,
        nullptr, 0, 0);
    rc |= frt_runtime_builder_add_port(
        b, "state", FRT_RT_MOD_STATE, FRT_RT_DTYPE_F32, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1, owner->state_shape, 1, 0,
        nullptr, 0, 0);
    rc |= frt_runtime_builder_add_port(
        b, "actions", FRT_RT_MOD_ACTION, FRT_RT_DTYPE_F32, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0, owner->action_shape, 2, 0,
        nullptr, 0, 0);
    rc |= frt_runtime_builder_add_callback_stage_v2(
        b, "infer", 0, nullptr, 0);
    rc |= frt_runtime_builder_add_identity(b, "provider", "llama_cpp");
    rc |= frt_runtime_builder_add_identity(b, "model_family", "pi0");
    rc |= frt_runtime_builder_add_identity(b, "model_path", config->model_path);
    rc |= frt_runtime_builder_add_identity(b, "mmproj_path", config->mmproj_path);
    rc |= frt_runtime_builder_add_identity(b, "backend", config->backend);
    if (rc != 0) {
        frt_runtime_builder_discard(b);
        destroy_owner(owner);
        return -1;
    }

    frt_model_runtime_verbs_v2 verbs{};
    verbs.struct_size = sizeof(verbs);
    verbs.set_input = set_input;
    verbs.get_output = get_output;
    verbs.prepare = unsupported_prepare;
    verbs.step = step;
    verbs.last_error = last_error;
    verbs.run_stage = run_stage;

    frt_model_runtime_v2* model = frt_runtime_builder_finish_model_v2(
        b, &verbs, owner, owner, nullptr, destroy_owner);
    if (!model) {
        destroy_owner(owner);
        return -1;
    }
    *out = model;
    return 0;
}
