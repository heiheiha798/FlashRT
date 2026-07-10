#include "flashrt/providers/llama_cpp/c_api.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

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

struct FakeFactory {
    FakeEngine* engine = nullptr;
    int creates = 0;
    frt_llama_cpp_pi0_config seen{};
    std::string seen_model_path;
    std::string seen_mmproj_path;
    std::string seen_backend;
    const char* last_error = "";
    bool return_borrowed = false;
    bool return_retain_only = false;
    bool return_undersized = false;
    bool return_null_self = false;
    bool return_missing_set_input = false;
    bool fail_after_engine = false;
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

int create_pi0_engine(void* p, const frt_llama_cpp_pi0_config* config,
                      frt_llama_cpp_engine_v1* out) {
    auto* factory = static_cast<FakeFactory*>(p);
    factory->creates += 1;
    if (!config || !out || !factory->engine) {
        factory->last_error = "invalid factory input";
        return -1;
    }
    if (factory->fail_after_engine) {
        out->struct_size = sizeof(*out);
        out->self = factory->engine;
        out->retain = retain_engine;
        out->release = release_engine;
        factory->last_error = "factory failed";
        return -7;
    }
    factory->seen = *config;
    factory->seen_model_path = config->model_path ? config->model_path : "";
    factory->seen_mmproj_path = config->mmproj_path ? config->mmproj_path : "";
    factory->seen_backend = config->backend ? config->backend : "";
    out->struct_size = factory->return_undersized ? 8u : sizeof(*out);
    out->self = factory->return_null_self ? nullptr : factory->engine;
    out->retain = (factory->return_borrowed ? nullptr : retain_engine);
    out->release = (factory->return_borrowed || factory->return_retain_only)
                       ? nullptr
                       : release_engine;
    out->set_input = factory->return_missing_set_input ? nullptr : set_input;
    out->run_infer = run_infer;
    out->get_output = get_output;
    out->last_error = last_error;
    return 0;
}

const char* factory_last_error(void* p) {
    return static_cast<FakeFactory*>(p)->last_error;
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

    FakeEngine old_engine;
    frt_llama_cpp_engine_v1 old_engine_api = engine_api;
    old_engine_api.self = &old_engine;
    old_engine_api.struct_size = FRT_LLAMA_CPP_ENGINE_V1_BASE_SIZE;
    old_engine_api.run_stage = nullptr;

    frt_model_runtime_v2* old_pi0 = nullptr;

    frt_llama_cpp_llm_config llm_cfg{};
    llm_cfg.struct_size = sizeof(llm_cfg);
    llm_cfg.model_path = "/models/llm.gguf";
    llm_cfg.backend = "cpu";
    llm_cfg.n_ctx = 2048;
    llm_cfg.max_tokens = 16;
    frt_model_runtime_v2* old_llm = nullptr;
    CHECK(frt_llama_cpp_llm_runtime_create_with_engine(
              &llm_cfg, &old_engine_api, &old_llm) == 0 && old_llm &&
              old_llm->n_ports == 2 && old_llm->n_stages_v2 == 1,
          "old-prefix LLM engine exposes only infer schema");
    if (old_llm) old_llm->release(old_llm->owner);

    frt_llama_cpp_mllm_config mllm_cfg{};
    mllm_cfg.struct_size = sizeof(mllm_cfg);
    mllm_cfg.model_path = "/models/mllm.gguf";
    mllm_cfg.mmproj_path = "/models/mllm-mmproj.gguf";
    mllm_cfg.backend = "cpu";
    mllm_cfg.n_ctx = 2048;
    mllm_cfg.max_tokens = 16;
    frt_model_runtime_v2* old_mllm = nullptr;
    CHECK(frt_llama_cpp_mllm_runtime_create_with_engine(
              &mllm_cfg, &old_engine_api, &old_mllm) == 0 && old_mllm &&
              old_mllm->n_ports == 3 && old_mllm->n_stages_v2 == 1,
          "old-prefix MLLM engine exposes only infer schema");
    if (old_mllm) old_mllm->release(old_mllm->owner);

    frt_llama_cpp_pi0_config cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.model_path = "/models/pi0.gguf";
    cfg.mmproj_path = "/models/pi0-mmproj.gguf";
    cfg.backend = "cpu";
    cfg.n_views = 2;
    cfg.image_height = 224;
    cfg.image_width = 224;
    cfg.image_channels = 3;
    cfg.action_steps = 2;
    cfg.action_dim = 2;

    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(
              &cfg, &old_engine_api, &old_pi0) == 0 && old_pi0 &&
              old_pi0->n_stages_v2 == 1,
          "old-prefix Pi0 engine exposes only infer schema");
    if (old_pi0) old_pi0->release(old_pi0->owner);

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
              model->ports[FRT_LLAMA_CPP_PI0_PORT_STATE].shape[0] == 2,
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

    FakeEngine factory_engine;
    FakeFactory factory;
    factory.engine = &factory_engine;
    frt_llama_cpp_engine_factory_v1 factory_api{};
    factory_api.struct_size = sizeof(factory_api);
    factory_api.self = &factory;
    factory_api.create_pi0 = create_pi0_engine;
    factory_api.last_error = factory_last_error;

    const char* open_json =
        "{"
        "\"model_family\":\"pi0\","
        "\"model_path\":\"/models/pi0.gguf\","
        "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
        "\"backend\":\"cpu\","
        "\"n_views\":2,"
        "\"image_height\":224,"
        "\"image_width\":224,"
        "\"image_channels\":3,"
        ""
        "\"action_steps\":2,"
        "\"action_dim\":2"
        "}";
    frt_model_runtime_v2* opened = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json, &factory_api, &opened) == 0 &&
              opened && factory.creates == 1,
          "open_with_engine_factory creates a Pi0 runtime from JSON");
    CHECK(factory.seen_model_path == "/models/pi0.gguf" &&
              factory.seen_mmproj_path == "/models/pi0-mmproj.gguf" &&
              factory.seen_backend == "cpu" &&
              factory.seen.n_views == 2 &&
              factory.seen.action_steps == 2 &&
              factory.seen.action_dim == 2,
          "factory receives parsed Pi0 config");
    CHECK(factory_engine.retains == 1 && factory_engine.releases == 1,
          "factory engine reference is transferred to the runtime");
    CHECK(opened->verbs_v2.run_stage(
              opened->self, FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER, -1) == 0 &&
              factory_engine.infer == 1,
          "opened runtime delegates infer to factory engine");
    opened->release(opened->owner);
    CHECK(factory_engine.releases == 2,
          "opened runtime releases retained factory engine");

    frt_model_runtime_v2* missing = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_path\":\"/models/pi0.gguf\"}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects incomplete JSON config");
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":2,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2,\"unexpected\":1}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects unknown JSON fields");
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_family\":\"pi0\","
              "\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":2,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects duplicate JSON fields");
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"llm\",\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":2,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects non-Pi0 model family");
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":0,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects zero-sized numeric config fields");
    FakeFactory borrowed_factory = factory;
    borrowed_factory.engine = &factory_engine;
    borrowed_factory.return_borrowed = true;
    frt_llama_cpp_engine_factory_v1 borrowed_factory_api = factory_api;
    borrowed_factory_api.self = &borrowed_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json, &borrowed_factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects borrowed engines from factories");
    const int releases_before_invalid_engine = factory_engine.releases;
    FakeFactory invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.return_undersized = true;
    frt_llama_cpp_engine_factory_v1 invalid_factory_api = factory_api;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json, &invalid_factory_api, &missing) == -1 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine,
          "open rejects undersized factory engines without calling release");
    invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.return_null_self = true;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json, &invalid_factory_api, &missing) == -1 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine,
          "open rejects null factory engine self without calling release");
    invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.return_retain_only = true;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json, &invalid_factory_api, &missing) == -1 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine,
          "open rejects asymmetric factory engines without calling release");
    invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.return_missing_set_input = true;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json, &invalid_factory_api, &missing) == -1 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine,
          "open rejects factory engines missing hot-path hooks without release");
    invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.fail_after_engine = true;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json, &invalid_factory_api, &missing) == -7 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine,
          "open ignores out_engine when factory create fails");

    std::printf(g_fail ? "\n== LLAMA_CPP PROVIDER FAILED ==\n"
                       : "\n== LLAMA_CPP PROVIDER PASSED ==\n");
    return g_fail;
}
