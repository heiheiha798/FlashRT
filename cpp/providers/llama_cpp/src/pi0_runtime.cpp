#include "flashrt/providers/llama_cpp/c_api.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
        !config->image_channels ||
        !config->action_steps || !config->action_dim) {
        return -1;
    }
    if (!engine || engine->struct_size < FRT_LLAMA_CPP_ENGINE_V1_BASE_SIZE ||
        !engine->self || !engine->set_input || !engine->run_infer ||
        !engine->get_output || !engine->last_error ||
        static_cast<bool>(engine->retain) !=
            static_cast<bool>(engine->release)) {
        return -1;
    }

    auto* owner = new (std::nothrow) RuntimeOwner();
    if (!owner) return -5;
    std::memcpy(&owner->engine, engine,
                std::min<size_t>(engine->struct_size, sizeof(owner->engine)));
    if (owner->engine.retain) owner->engine.retain(owner->engine.self);

    owner->image_shape[0] = static_cast<int64_t>(config->n_views);
    owner->image_shape[1] = static_cast<int64_t>(config->image_height);
    owner->image_shape[2] = static_cast<int64_t>(config->image_width);
    owner->image_shape[3] = static_cast<int64_t>(config->image_channels);
    // Pi0 state shares the model action_dim: llama_set_pi0_state requires
    // n_values == hparams.action_dim, and the caller zero-pads real
    // proprioception into that width. Exposing it on the state port keeps
    // the host-visible shape honest.
    owner->state_shape[0] = static_cast<int64_t>(config->action_dim);
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
    // Phase 6: the actions OUT port carries a memory-domain token (handle =
    // engine.self, the provider Engine*) so a consumer can read the action
    // chunk through frt_memory_token_verbs as well as get_output. HOST_VISIBLE:
    // actions_buf is host memory, read live at copy_to_host call time. The
    // port desc this produces is byte-identical to add_port (STAGED, no raw
    // buffer), so the model identity/fingerprint and get_output are unchanged.
    // Only minted when a real Jetson-PI engine is linked; without
    // FLASHRT_CPP_WITH_JETSON_PI there is no backing store to advertise (the
    // fake-engine unit test path uses add_port and reads via get_output).
#if defined(FLASHRT_CPP_WITH_JETSON_PI)
    const uint64_t action_window_bytes =
        static_cast<uint64_t>(config->action_steps) *
        static_cast<uint64_t>(config->action_dim) * sizeof(float);
    rc |= frt_runtime_builder_add_port_token(
        b, "actions", FRT_RT_MOD_ACTION, FRT_RT_DTYPE_F32, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_OUT, 0, owner->action_shape, 2, 0,
        reinterpret_cast<frt_memory_token>(owner->engine.self),
        frt_jetson_pi_actions_token_verbs(),
        /*offset=*/0, action_window_bytes, FRT_RT_LOCATION_HOST_VISIBLE);
#else
    rc |= frt_runtime_builder_add_port(
        b, "actions", FRT_RT_MOD_ACTION, FRT_RT_DTYPE_F32, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0, owner->action_shape, 2, 0,
        nullptr, 0, 0);
#endif
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

extern "C" int frt_llama_cpp_pi0_runtime_open_with_engine_factory(
        const char* config_json,
        const frt_llama_cpp_engine_factory_v1* factory,
        frt_model_runtime_v2** out) {
    if (!out) return -1;
    *out = nullptr;
    if (!factory ||
        factory->struct_size < sizeof(frt_llama_cpp_engine_factory_v1) ||
        !factory->create_pi0 || !factory->last_error) {
        return -1;
    }
    if (!config_json) {
        return -1;
    }

    frt_llama_cpp_pi0_config config{};
    config.struct_size = sizeof(config);
    std::string model_family;
    std::string model_path;
    std::string mmproj_path;
    std::string backend;
    bool seen_model_family = false;
    bool seen_model_path = false;
    bool seen_mmproj_path = false;
    bool seen_backend = false;
    bool seen_n_views = false;
    bool seen_image_height = false;
    bool seen_image_width = false;
    bool seen_image_channels = false;
    bool seen_action_steps = false;
    bool seen_action_dim = false;
    const char* p = config_json;
    auto skip_ws = [&p]() {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    };
    auto parse_string = [&p](std::string* out) {
        if (*p != '"') return false;
        ++p;
        out->clear();
        while (*p && *p != '"') {
            if (*p == '\\') {
                ++p;
                switch (*p) {
                    case '"':
                    case '\\':
                    case '/':
                        out->push_back(*p++);
                        break;
                    case 'b':
                        out->push_back('\b');
                        ++p;
                        break;
                    case 'f':
                        out->push_back('\f');
                        ++p;
                        break;
                    case 'n':
                        out->push_back('\n');
                        ++p;
                        break;
                    case 'r':
                        out->push_back('\r');
                        ++p;
                        break;
                    case 't':
                        out->push_back('\t');
                        ++p;
                        break;
                    default:
                        return false;
                }
            } else {
                const unsigned char ch = static_cast<unsigned char>(*p);
                if (ch < 0x20) return false;
                out->push_back(*p++);
            }
        }
        if (*p != '"') return false;
        ++p;
        return true;
    };
    auto parse_u32 = [&p](uint32_t* out) {
        uint32_t value = 0;
        if (*p < '0' || *p > '9') return false;
        if (*p == '0') {
            ++p;
        } else {
            while (*p >= '0' && *p <= '9') {
                const uint32_t digit = static_cast<uint32_t>(*p - '0');
                if (value > (UINT32_MAX - digit) / 10u) return false;
                value = value * 10u + digit;
                ++p;
            }
        }
        if (value == 0) return false;
        *out = value;
        return true;
    };
    skip_ws();
    if (*p != '{') return -1;
    ++p;
    skip_ws();
    if (*p == '}') return -1;
    for (;;) {
        std::string key;
        if (!parse_string(&key)) return -1;
        skip_ws();
        if (*p != ':') return -1;
        ++p;
        skip_ws();
        if (key == "model_family") {
            if (seen_model_family || !parse_string(&model_family) ||
                model_family.empty()) {
                return -1;
            }
            seen_model_family = true;
        } else if (key == "model_path") {
            if (seen_model_path || !parse_string(&model_path) ||
                model_path.empty()) {
                return -1;
            }
            seen_model_path = true;
        } else if (key == "mmproj_path") {
            if (seen_mmproj_path || !parse_string(&mmproj_path) ||
                mmproj_path.empty()) {
                return -1;
            }
            seen_mmproj_path = true;
        } else if (key == "backend") {
            if (seen_backend || !parse_string(&backend) || backend.empty()) {
                return -1;
            }
            seen_backend = true;
        } else if (key == "n_views") {
            if (seen_n_views || !parse_u32(&config.n_views)) return -1;
            seen_n_views = true;
        } else if (key == "image_height") {
            if (seen_image_height || !parse_u32(&config.image_height)) {
                return -1;
            }
            seen_image_height = true;
        } else if (key == "image_width") {
            if (seen_image_width || !parse_u32(&config.image_width)) return -1;
            seen_image_width = true;
        } else if (key == "image_channels") {
            if (seen_image_channels || !parse_u32(&config.image_channels)) {
                return -1;
            }
            seen_image_channels = true;
        } else if (key == "action_steps") {
            if (seen_action_steps || !parse_u32(&config.action_steps)) {
                return -1;
            }
            seen_action_steps = true;
        } else if (key == "action_dim") {
            if (seen_action_dim || !parse_u32(&config.action_dim)) return -1;
            seen_action_dim = true;
        } else {
            return -1;
        }
        skip_ws();
        if (*p == '}') {
            ++p;
            break;
        }
        if (*p != ',') return -1;
        ++p;
        skip_ws();
    }
    skip_ws();
    if (*p != '\0') return -1;
    if (!seen_model_family || model_family != "pi0" || !seen_model_path ||
        !seen_mmproj_path || !seen_backend || !seen_n_views ||
        !seen_image_height || !seen_image_width || !seen_image_channels ||
        !seen_action_steps || !seen_action_dim) {
        return -1;
    }
    config.model_path = model_path.c_str();
    config.mmproj_path = mmproj_path.c_str();
    config.backend = backend.c_str();

    frt_llama_cpp_engine_v1 engine{};
    const int rc = factory->create_pi0(factory->self, &config, &engine);
    if (rc != 0) {
        return rc;
    }
    if (engine.struct_size < FRT_LLAMA_CPP_ENGINE_V1_BASE_SIZE ||
        !engine.self || !engine.retain || !engine.release ||
        !engine.set_input || !engine.run_infer || !engine.get_output ||
        !engine.last_error) {
        return -1;
    }
    frt_model_runtime_v2* model = nullptr;
    const int create_rc =
        frt_llama_cpp_pi0_runtime_create_with_engine(&config, &engine,
                                                     &model);
    engine.release(engine.self);
    if (create_rc != 0) return create_rc;
    *out = model;
    return 0;
}
