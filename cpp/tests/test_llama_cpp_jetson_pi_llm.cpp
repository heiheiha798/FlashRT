// End-to-end LLM engine test: drives the real Jetson-PI LLM engine factory
// with a GGUF LLM (when available) and asserts a sane text completion.
// Skips (returns 0) when FLASHRT_LLM_MODEL is unset.

#include "flashrt/providers/llama_cpp/c_api.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

int main() {
    const char * model_env = std::getenv("FLASHRT_LLM_MODEL");
    if (!model_env || !*model_env || !std::fopen(model_env, "rb")) {
        std::printf("SKIP - FLASHRT_LLM_MODEL not set or file missing\n");
        return 0;
    }

    const frt_llama_cpp_engine_factory_v1 * factory =
        frt_llama_cpp_default_engine_factory();
    CHECK(factory != nullptr && factory->create_llm != nullptr,
          "default engine factory exposes create_llm");

    // Bogus model path must fail cleanly.
    const char * bogus_json =
        "{\"model_family\":\"llm\","
        "\"model_path\":\"/nonexistent/bogus.gguf\","
        "\"backend\":\"cpu\","
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.8,\"top_k\":40,\"top_p\":0.9,\"seed\":1,\"max_tokens\":64}";
    frt_model_runtime_v2 * bogus = nullptr;
    CHECK(frt_llama_cpp_llm_runtime_open_with_engine_factory(
              bogus_json, factory, &bogus) != 0 && bogus == nullptr,
          "open LLM with bogus model path fails without crashing");

    // Real model.
    const char * prompt = "What is 2 plus 2? The answer is";
    std::string json =
        std::string("{") +
        "\"model_family\":\"llm\","
        "\"model_path\":\"" + model_env + "\","
        "\"backend\":\"cpu\","
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.0,\"top_k\":0,\"top_p\":0.0,\"seed\":1,\"max_tokens\":64}";
    frt_model_runtime_v2 * model = nullptr;
    int rc = frt_llama_cpp_llm_runtime_open_with_engine_factory(
        json.c_str(), factory, &model);
    if (rc != 0 || !model) {
        std::printf("FAIL: open LLM runtime (rc=%d): %s\n", rc,
                    factory->last_error(factory->self));
        return 1;
    }
    CHECK(rc == 0 && model != nullptr, "open LLM runtime from JSON");

    rc = model->verbs_v2.set_input(model->self,
                                   FRT_LLAMA_CPP_LLM_PORT_PROMPT,
                                   prompt, std::strlen(prompt), -1);
    CHECK(rc == 0, "set_input prompt");

    rc = model->verbs_v2.run_stage(
        model->self, FRT_LLAMA_CPP_LLM_STAGE_INDEX_INFER, -1);
    if (rc != 0) {
        std::printf("FAIL: run_stage infer (rc=%d): %s\n", rc,
                    model->verbs_v2.last_error(model->self));
        g_fail = 1;
    } else {
        std::printf("ok  : run_stage infer\n");
    }

    // get_output: read directly into a generously-sized buffer (the engine
    // reports the worst-case max_tokens*8 on size-query, so a size-query
    // first would force an oversized alloc). Allocate worst-case, read,
    // then trim.
    const uint64_t cap = 64u * 8u;  // max_tokens(64) * 8 worst-case bytes
    std::string text(cap, '\0');
    uint64_t written = 0;
    rc = model->verbs_v2.get_output(
        model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT, &text[0], text.size(),
        &written, -1);
    CHECK(rc == 0 && written > 0, "get_output text non-empty");
    if (rc == 0) {
        text.resize(written);
        std::printf("    generated: %s\n", text.c_str());
        // Sanity: contains at least one printable character.
        bool printable = false;
        for (char c : text) {
            if (c >= 0x20 && c < 0x7f) { printable = true; break; }
        }
        CHECK(printable, "generated text contains printable chars");
    }

    model->release(model->owner);

    std::printf(g_fail ? "\n== JETSON_PI LLM FAILED ==\n"
                       : "\n== JETSON_PI LLM PASSED ==\n");
    return g_fail;
}
