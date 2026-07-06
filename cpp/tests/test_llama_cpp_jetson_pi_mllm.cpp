// End-to-end MLLM engine test: drives the real Jetson-PI MLLM engine factory
// with a GGUF VLM (when available) + a programmatically generated test image,
// and asserts a sane text completion. Skips (returns 0) when env vars unset.
//
// Env:
//   FLASHRT_MLLM_MODEL     path to VLM GGUF (e.g. Qwen2.5-VL-3B-Instruct-q4_0.gguf)
//   FLASHRT_MLLM_MMPROJ    path to VIT mmproj GGUF

#include "flashrt/providers/llama_cpp/c_api.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

int main() {
    const char * model_env = std::getenv("FLASHRT_MLLM_MODEL");
    const char * mmproj_env = std::getenv("FLASHRT_MLLM_MMPROJ");
    auto file_exists = [](const char * path) {
        if (!path || !*path) return false;
        FILE * f = std::fopen(path, "rb");
        if (!f) return false;
        std::fclose(f);
        return true;
    };
    if (!file_exists(model_env) || !file_exists(mmproj_env)) {
        std::printf("SKIP - FLASHRT_MLLM_MODEL / FLASHRT_MLLM_MMPROJ not set or missing\n");
        return 0;
    }

    const frt_llama_cpp_engine_factory_v1 * factory =
        frt_llama_cpp_default_engine_factory();
    CHECK(factory != nullptr && factory->create_mllm != nullptr,
          "default engine factory exposes create_mllm");

    // Bogus model path must fail cleanly.
    const char * bogus_json =
        "{\"model_family\":\"mllm\","
        "\"model_path\":\"/nonexistent/bogus.gguf\","
        "\"mmproj_path\":\"/nonexistent/bogus-mmproj.gguf\","
        "\"backend\":\"cpu\","
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.0,\"top_k\":0,\"top_p\":0.0,\"seed\":1,\"max_tokens\":32}";
    frt_model_runtime_v2 * bogus = nullptr;
    CHECK(frt_llama_cpp_mllm_runtime_open_with_engine_factory(
              bogus_json, factory, &bogus) != 0 && bogus == nullptr,
          "open MLLM with bogus model path fails without crashing");

    // Real VLM.
    std::string json =
        std::string("{") +
        "\"model_family\":\"mllm\","
        "\"model_path\":\"" + model_env + "\","
        "\"mmproj_path\":\"" + mmproj_env + "\","
        "\"backend\":\"cpu\","
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.0,\"top_k\":0,\"top_p\":0.0,\"seed\":1,\"max_tokens\":64}";
    frt_model_runtime_v2 * model = nullptr;
    int rc = frt_llama_cpp_mllm_runtime_open_with_engine_factory(
        json.c_str(), factory, &model);
    if (rc != 0 || !model) {
        std::printf("FAIL: open MLLM runtime (rc=%d): %s\n", rc,
                    factory->last_error(factory->self));
        return 1;
    }
    CHECK(rc == 0 && model != nullptr, "open MLLM runtime from JSON");

    // Generate a 224x224 solid-red RGB image inline (no stb_image_write dep).
    const int W = 224, H = 224;
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0; i < rgb.size(); i += 3) {
        rgb[i] = 255; rgb[i + 1] = 0; rgb[i + 2] = 0;
    }

    frt_image_view view;
    view.struct_size = sizeof(frt_image_view);
    view.pixel_format = FRT_RT_PIXEL_RGB8;
    view.data = rgb.data();
    view.bytes = rgb.size();
    view.width = W; view.height = H; view.stride_bytes = 0;
    view.reserved = 0; view.timestamp_ns = 0;

    rc = model->verbs_v2.set_input(model->self,
                                   FRT_LLAMA_CPP_MLLM_PORT_IMAGES,
                                   &view, sizeof(view), -1);
    CHECK(rc == 0, "set_input images");

    const char * prompt = "Describe this image in one sentence.";
    rc = model->verbs_v2.set_input(model->self,
                                   FRT_LLAMA_CPP_MLLM_PORT_PROMPT,
                                   prompt, std::strlen(prompt), -1);
    CHECK(rc == 0, "set_input prompt");

    rc = model->verbs_v2.run_stage(
        model->self, FRT_LLAMA_CPP_MLLM_STAGE_INDEX_INFER, -1);
    if (rc != 0) {
        std::printf("FAIL: run_stage infer (rc=%d): %s\n", rc,
                    model->verbs_v2.last_error(model->self));
        g_fail = 1;
    } else {
        std::printf("ok  : run_stage infer\n");
    }

    const uint64_t cap = 64u * 8u;
    std::string text(cap, '\0');
    uint64_t written = 0;
    rc = model->verbs_v2.get_output(
        model->self, FRT_LLAMA_CPP_MLLM_PORT_TEXT, &text[0], text.size(),
        &written, -1);
    CHECK(rc == 0 && written > 0, "get_output text non-empty");
    if (rc == 0) {
        text.resize(written);
        std::printf("    generated: %s\n", text.c_str());
        bool printable = false;
        for (char c : text) {
            if (c >= 0x20 && c < 0x7f) { printable = true; break; }
        }
        CHECK(printable, "generated text contains printable chars");
    }

    model->release(model->owner);

    std::printf(g_fail ? "\n== JETSON_PI MLLM FAILED ==\n"
                       : "\n== JETSON_PI MLLM PASSED ==\n");
    return g_fail;
}
