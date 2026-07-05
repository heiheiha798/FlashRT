#include "flashrt/providers/llama_cpp/c_api.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

namespace {

struct FakeEngine {
    int retains = 0;
    int releases = 0;
    int infer = 0;
    int set_input_calls = 0;
    uint32_t last_port = 999;
    float actions[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const char* last_error = "";
};

void retain_engine(void* p) {
    static_cast<FakeEngine*>(p)->retains += 1;
}

void release_engine(void* p) {
    static_cast<FakeEngine*>(p)->releases += 1;
}

int set_input(void* p, uint32_t port, const void* data, uint64_t bytes,
              int stream) {
    (void)data;
    (void)bytes;
    (void)stream;
    auto* engine = static_cast<FakeEngine*>(p);
    engine->set_input_calls += 1;
    engine->last_port = port;
    return 0;
}

int run_infer(void* p) {
    auto* engine = static_cast<FakeEngine*>(p);
    engine->infer += 1;
    engine->actions[0] += 10.0f;
    return 0;
}

int get_output(void* p, uint32_t port, void* out, uint64_t capacity,
               uint64_t* written, int stream) {
    (void)stream;
    auto* engine = static_cast<FakeEngine*>(p);
    if (port != FRT_LLAMA_CPP_PI0_PORT_ACTIONS) {
        engine->last_error = "unknown output port";
        return -1;
    }
    const uint64_t need = sizeof(engine->actions);
    if (written) *written = need;
    if (capacity < need) {
        engine->last_error = "action output buffer is too small";
        return -5;
    }
    std::memcpy(out, engine->actions, need);
    engine->last_error = "";
    return 0;
}

const char* last_error(void* p) {
    return static_cast<FakeEngine*>(p)->last_error;
}

const char* null_last_error(void*) {
    return nullptr;
}

}  // namespace

int main() {
    frt_model_runtime_v2* bad = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(nullptr, nullptr, &bad)
              == -1 && bad == nullptr,
          "create rejects missing config and engine");

    FakeEngine engine;
    frt_llama_cpp_engine_v1 engine_api{};
    engine_api.struct_size = sizeof(engine_api);
    engine_api.self = &engine;
    engine_api.retain = retain_engine;
    engine_api.release = release_engine;
    engine_api.set_input = set_input;
    engine_api.run_infer = run_infer;
    engine_api.get_output = get_output;
    engine_api.last_error = last_error;

    frt_llama_cpp_pi0_config cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.model_path = "/models/pi0.gguf";
    cfg.mmproj_path = "/models/pi0-mmproj.gguf";
    cfg.backend = "cpu";
    cfg.n_views = 2;
    cfg.image_height = 224;
    cfg.image_width = 224;
    cfg.image_channels = 3;
    cfg.state_dim = 8;
    cfg.action_steps = 2;
    cfg.action_dim = 2;

    frt_llama_cpp_engine_v1 asymmetric = engine_api;
    asymmetric.retain = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &asymmetric,
                                                       &bad) == -1 &&
              bad == nullptr,
          "create rejects release without retain");

    asymmetric = engine_api;
    asymmetric.release = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &asymmetric,
                                                       &bad) == -1 &&
              bad == nullptr,
          "create rejects retain without release");

    frt_llama_cpp_engine_v1 borrowed = engine_api;
    borrowed.retain = nullptr;
    borrowed.release = nullptr;
    frt_model_runtime_v2* borrowed_model = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &borrowed,
                                                       &borrowed_model) == 0 &&
              borrowed_model,
          "create accepts a borrowed explicit engine");
    borrowed_model->release(borrowed_model->owner);

    frt_model_runtime_v2* model = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &engine_api,
                                                       &model) == 0 && model,
          "create llama_cpp Pi0 runtime with explicit engine");
    CHECK(engine.retains == 1, "engine retained once");
    CHECK(model->abi_version == FRT_MODEL_RUNTIME_ABI_VERSION_V2 &&
              model->n_ports == 4 && model->n_stages == 0 &&
              model->n_stages_v2 == 1,
          "runtime exposes v2 provider-owned shape");
    CHECK(model->exp && model->exp->ctx == nullptr &&
              model->exp->n_graphs == 0,
          "provider runtime has no FlashRT exec graph");
    CHECK(std::strcmp(model->ports[FRT_LLAMA_CPP_PI0_PORT_IMAGES].name,
                      "images") == 0 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_IMAGES].update ==
                  FRT_RT_PORT_STAGED &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_IMAGES].shape[0] == 2 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_IMAGES].shape[3] == 3,
          "images port schema");
    CHECK(std::strcmp(model->ports[FRT_LLAMA_CPP_PI0_PORT_PROMPT].name,
                      "prompt") == 0 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_PROMPT].modality ==
                  FRT_RT_MOD_TEXT &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_PROMPT].shape[0] == -1,
          "prompt port schema");
    CHECK(std::strcmp(model->ports[FRT_LLAMA_CPP_PI0_PORT_STATE].name,
                      "state") == 0 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_STATE].dtype ==
                  FRT_RT_DTYPE_F32 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_STATE].shape[0] == 8,
          "state port schema");
    CHECK(std::strcmp(model->ports[FRT_LLAMA_CPP_PI0_PORT_ACTIONS].name,
                      "actions") == 0 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_ACTIONS].direction ==
                  FRT_RT_PORT_OUT &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_ACTIONS].shape[0] == 2 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_ACTIONS].shape[1] == 2,
          "actions port schema");
    CHECK(std::strcmp(model->stages_v2[0].name, "infer") == 0 &&
              model->stages_v2[0].kind == FRT_RT_STAGE_CALLBACK &&
              model->stages_v2[0].callback == 0,
          "infer callback stage schema");
    CHECK(std::strstr(model->exp->identity, "provider=llama_cpp\n") &&
              std::strstr(model->exp->identity, "model_family=pi0\n") &&
              std::strstr(model->exp->identity, "stage_v2:0:infer:"),
          "identity carries provider, model family, and callback stage");

    CHECK(model->verbs_v2.set_input(model->self, FRT_LLAMA_CPP_PI0_PORT_PROMPT,
                                    "hello", 5, -1) == 0 &&
              engine.set_input_calls == 1 &&
              engine.last_port == FRT_LLAMA_CPP_PI0_PORT_PROMPT,
          "set_input delegates to engine");
    CHECK(model->verbs_v2.run_stage(
              model->self, FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER, -1) == 0 &&
              engine.infer == 1,
          "run_stage delegates infer to engine");
    float out[4] = {};
    uint64_t written = 0;
    CHECK(model->verbs_v2.get_output(model->self,
                                     FRT_LLAMA_CPP_PI0_PORT_ACTIONS,
                                     out, sizeof(out), &written, -1) == 0 &&
              written == sizeof(out) && std::fabs(out[0] - 11.0f) < 0.01f,
          "get_output delegates to engine");
    CHECK(model->verbs_v2.prepare(model->self, 0, 0) == -3 &&
              std::strstr(model->verbs_v2.last_error(model->self),
                          "no graph variants"),
          "prepare hard-errors instead of fabricating graph variants");
    model->release(model->owner);
    CHECK(engine.releases == 1, "engine released once");

    FakeEngine null_error_engine;
    frt_llama_cpp_engine_v1 null_error_api = engine_api;
    null_error_api.self = &null_error_engine;
    null_error_api.last_error = null_last_error;
    frt_model_runtime_v2* null_error_model = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(
              &cfg, &null_error_api, &null_error_model) == 0 &&
              null_error_model,
          "create accepts engine with nullable error implementation");
    CHECK(null_error_model->verbs_v2.get_output(
              null_error_model->self, FRT_LLAMA_CPP_PI0_PORT_PROMPT,
              out, sizeof(out), &written, -1) == -1 &&
              std::strstr(null_error_model->verbs_v2.last_error(
                              null_error_model->self),
                          "null error"),
          "null engine errors are reported without crashing");
    null_error_model->release(null_error_model->owner);

    std::printf(g_fail ? "\n== LLAMA_CPP PROVIDER FAILED ==\n"
                       : "\n== LLAMA_CPP PROVIDER PASSED ==\n");
    return g_fail;
}
