// End-to-end Pi0 engine test: drives the real Jetson-PI engine factory with
// the LIBERO Pi0 weights (when available) and asserts a sane action chunk.
// Skips (returns 0) when the weights/fixture env vars are unset, so CI without
// weights still passes.
//
// Env:
//   FLASHRT_PI0_MODEL        path to Pi0 policy GGUF
//   FLASHRT_PI0_MMPROJ       path to VIT mmproj GGUF
//   FLASHRT_PI0_FIXTURE_DIR  dir containing image.png, wrist_image.png,
//                            state.bin (action_dim float32), prompt.txt

#include "flashrt/providers/llama_cpp/c_api.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb/stb_image.h"

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

namespace {

bool file_exists(const char * p) {
    std::ifstream f(p);
    return f.good();
}

std::string read_file(const std::string & path, bool * ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (ok) *ok = false; return {}; }
    std::string s((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
    if (ok) *ok = true;
    return s;
}

} // namespace

int main() {
    const char * model_env = std::getenv("FLASHRT_PI0_MODEL");
    const char * mmproj_env = std::getenv("FLASHRT_PI0_MMPROJ");
    const char * fixture_env = std::getenv("FLASHRT_PI0_FIXTURE_DIR");
    if (!model_env || !mmproj_env || !fixture_env ||
        !file_exists(model_env) || !file_exists(mmproj_env)) {
        std::printf("SKIP - FLASHRT_PI0_MODEL / FLASHRT_PI0_MMPROJ / "
                    "FLASHRT_PI0_FIXTURE_DIR not set or files missing\n");
        return 0;
    }

    const std::string fixture_dir = fixture_env;
    const std::string img_path = fixture_dir + "/image.png";
    const std::string wrist_path = fixture_dir + "/wrist_image.png";
    const std::string state_path = fixture_dir + "/state.bin";
    const std::string prompt_path = fixture_dir + "/prompt.txt";
    if (!file_exists(img_path.c_str()) ||
        !file_exists(wrist_path.c_str()) ||
        !file_exists(state_path.c_str()) ||
        !file_exists(prompt_path.c_str())) {
        std::printf("SKIP - fixture files missing in %s\n", fixture_dir.c_str());
        return 0;
    }

    // ---- sub-test A: bogus model path fails (no-weights contract) ----------
    const frt_llama_cpp_engine_factory_v1 * factory =
        frt_llama_cpp_default_engine_factory();
    CHECK(factory != nullptr, "default engine factory is non-null");
    CHECK(factory->create_pi0 != nullptr && factory->last_error != nullptr,
          "factory vtable complete");

    const char * bogus_json =
        "{\"model_family\":\"pi0\","
        "\"model_path\":\"/nonexistent/bogus.gguf\","
        "\"mmproj_path\":\"/nonexistent/bogus-mmproj.gguf\","
        "\"backend\":\"cpu\",\"n_views\":2,\"image_height\":224,"
        "\"image_width\":224,\"image_channels\":3,"
        "\"action_steps\":10,\"action_dim\":32}";
    frt_model_runtime_v2 * bogus = nullptr;
    int rc = frt_llama_cpp_pi0_runtime_open_with_engine_factory(
        bogus_json, factory, &bogus);
    CHECK(rc != 0 && bogus == nullptr,
          "open with bogus model path fails without crashing");

    // ---- sub-test B: end-to-end Pi0 tick -----------------------------------
    // Action dims come from env (default 50x32 for LIBERO base; pi0_base is 10x32).
    const char * steps_env = std::getenv("FLASHRT_PI0_ACTION_STEPS");
    const char * dim_env   = std::getenv("FLASHRT_PI0_ACTION_DIM");
    const long action_steps = steps_env ? std::atol(steps_env) : 50;
    const long action_dim   = dim_env   ? std::atol(dim_env)   : 32;
    if (action_steps <= 0 || action_dim <= 0 || action_steps > 10000 ||
        action_dim > 10000) {
        std::printf("SKIP - bad FLASHRT_PI0_ACTION_STEPS/DIM\n");
        return 0;
    }
    std::string json =
        std::string("{") +
        "\"model_family\":\"pi0\","
        "\"model_path\":\"" + model_env + "\","
        "\"mmproj_path\":\"" + mmproj_env + "\","
        "\"backend\":\"cpu\","
        "\"n_views\":2,\"image_height\":224,\"image_width\":224,"
        "\"image_channels\":3,\"action_steps\":" + std::to_string(action_steps) +
        ",\"action_dim\":" + std::to_string(action_dim) + "}";
    frt_model_runtime_v2 * model = nullptr;
    rc = frt_llama_cpp_pi0_runtime_open_with_engine_factory(
        json.c_str(), factory, &model);
    if (rc != 0 || !model) {
        std::printf("FAIL: open real Pi0 runtime (rc=%d): %s\n", rc,
                    factory->last_error(factory->self));
        return 1;
    }
    CHECK(rc == 0 && model != nullptr, "open real Pi0 runtime from JSON");

    // Load fixture.
    int iw = 0, ih = 0, ic = 0;
    unsigned char * img = stbi_load(img_path.c_str(), &iw, &ih, &ic, 3);
    CHECK(img != nullptr && iw == 224 && ih == 224, "load image.png 224x224");
    int ww = 0, wh = 0, wc = 0;
    unsigned char * wrist = stbi_load(wrist_path.c_str(), &ww, &wh, &wc, 3);
    CHECK(wrist != nullptr && ww == 224 && wh == 224, "load wrist_image.png 224x224");
    bool state_ok = false;
    std::string state_bytes = read_file(state_path, &state_ok);
    CHECK(state_ok && state_bytes.size() ==
                      static_cast<size_t>(action_dim) * sizeof(float),
          "state.bin matches action_dim float32");
    bool prompt_ok = false;
    std::string prompt = read_file(prompt_path, &prompt_ok);
    CHECK(prompt_ok && !prompt.empty(), "prompt.txt non-empty");
    if (!prompt.empty() && prompt.back() == '\n') prompt.pop_back();

    if (img && wrist && state_ok && prompt_ok) {
        frt_image_view views[2];
        views[0].struct_size = sizeof(frt_image_view);
        views[0].pixel_format = FRT_RT_PIXEL_RGB8;
        views[0].data = img;
        views[0].bytes = static_cast<uint64_t>(224) * 224 * 3;
        views[0].width = 224; views[0].height = 224; views[0].stride_bytes = 0;
        views[0].reserved = 0; views[0].timestamp_ns = 0;
        views[1] = views[0];
        views[1].data = wrist;

        CHECK(model->verbs_v2.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_IMAGES,
                                        &views[0], sizeof(views), -1) == 0,
              "set_input images");
        CHECK(model->verbs_v2.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_PROMPT,
                                        prompt.data(), prompt.size(), -1) == 0,
              "set_input prompt");
        CHECK(model->verbs_v2.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_STATE,
                                        state_bytes.data(),
                                        state_bytes.size(), -1) == 0,
              "set_input state");

        rc = model->verbs_v2.run_stage(
            model->self, FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER, -1);
        if (rc != 0) {
            std::printf("FAIL: run_stage infer (rc=%d): %s\n", rc,
                        model->verbs_v2.last_error(model->self));
            g_fail = 1;
        } else {
            std::printf("ok  : run_stage infer\n");
        }

        std::vector<float> actions(
            static_cast<size_t>(action_steps) * action_dim);
        uint64_t written = 0;
        rc = model->verbs_v2.get_output(
            model->self, FRT_LLAMA_CPP_PI0_PORT_ACTIONS,
            actions.data(), actions.size() * sizeof(float), &written, -1);
        CHECK(rc == 0 &&
                  written == static_cast<uint64_t>(action_steps) * action_dim *
                                 sizeof(float),
              "get_output actions shape matches config");
        if (rc == 0) {
            bool nan_inf = false, all_zero = true;
            for (float v : actions) {
                if (std::isnan(v) || std::isinf(v)) { nan_inf = true; break; }
                if (v != 0.0f) all_zero = false;
            }
            CHECK(!nan_inf, "actions contain no NaN/Inf");
            CHECK(!all_zero, "actions are not all zero");
        }
    }

    model->release(model->owner);
    if (img)   stbi_image_free(img);
    if (wrist) stbi_image_free(wrist);

    std::printf(g_fail ? "\n== JETSON_PI ENGINE FAILED ==\n"
                       : "\n== JETSON_PI ENGINE PASSED ==\n");
    return g_fail;
}
