// PI0.5 reference-policy state-parity scaffold.
//
// This test guards the PI0.5 proprioceptive-state contract that the narrow
// Jetson-PI C API does NOT yet honor: for PI0.5 the policy consumes the
// discretized state inside the `Task: ..., State: ...;\nAction:` prompt, not
// the legacy llama_set_pi0_state tensor (which the PI0.5 graph never reads).
// Until the Jetson-PI-Edge backend wires that serialization in, this test
// SKIPS. Once it lands, the test asserts that varying state while holding
// images/task fixed changes the action chunk — the regression that LiangSu
// reported on FlashRT #148 (zeros-vs-ones produced bit-identical actions).
//
// Boundary-region coverage matters: the openpi discretizer maps x<-1 to a
// literal -1 token (distinct from bin 0), x in [-1,1) to floor((x+1)*128),
// and x>=1 to 255. A wrong binning formula still passes a naive "changes when
// state changes" check, so each region is exercised explicitly.
//
// The caller is responsible for passing state already normalized into [-1,1]
// (matching the Jetson-PI server path, which bins pi0_req.state directly).
//
// Env:
//   FLASHRT_PI0_MODEL        path to PI0.5 policy GGUF
//   FLASHRT_PI0_MMPROJ       path to VIT mmproj GGUF
//   FLASHRT_PI0_FIXTURE_DIR  dir with image.png, wrist_image.png, prompt.txt
//   FLASHRT_PI0_ACTION_STEPS (optional) override; default 50 (LIBERO base).
//   FLASHRT_PI0_ACTION_DIM   (optional) override; default 32.
//   FLASHRT_PI0_BACKEND      (optional) "cpu" (default) or "cuda".
//   FLASHRT_PI0_STATE_PARITY_READY  must be set to "1" to opt in. Until the
//                            backend PI0.5 state serialization lands, this is
//                            intentionally unset so the test skips.

#include "flashrt/providers/llama_cpp/c_api.h"
#include "llama_cpp_generic_plan.h"
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
    const char * ready = std::getenv("FLASHRT_PI0_STATE_PARITY_READY");
    if (!ready || ready[0] != '1') {
        std::printf("SKIP - PI0.5 state parity awaits the Jetson-PI-Edge "
                    "backend serialization (set FLASHRT_PI0_STATE_PARITY_READY=1 "
                    "once it lands)\n");
        return 0;
    }
    const char * model_env   = std::getenv("FLASHRT_PI0_MODEL");
    const char * mmproj_env  = std::getenv("FLASHRT_PI0_MMPROJ");
    const char * fixture_env = std::getenv("FLASHRT_PI0_FIXTURE_DIR");
    if (!model_env || !mmproj_env || !fixture_env ||
        !file_exists(model_env) || !file_exists(mmproj_env)) {
        std::printf("SKIP - FLASHRT_PI0_MODEL / FLASHRT_PI0_MMPROJ / "
                    "FLASHRT_PI0_FIXTURE_DIR not set or files missing\n");
        return 0;
    }
    const std::string fixture_dir = fixture_env;
    const std::string img_path    = fixture_dir + "/image.png";
    const std::string wrist_path  = fixture_dir + "/wrist_image.png";
    const std::string prompt_path = fixture_dir + "/prompt.txt";
    if (!file_exists(img_path.c_str()) || !file_exists(wrist_path.c_str()) ||
        !file_exists(prompt_path.c_str())) {
        std::printf("SKIP - fixture files missing in %s\n", fixture_dir.c_str());
        return 0;
    }

    const char * steps_env = std::getenv("FLASHRT_PI0_ACTION_STEPS");
    const char * dim_env   = std::getenv("FLASHRT_PI0_ACTION_DIM");
    const char * be_env    = std::getenv("FLASHRT_PI0_BACKEND");
    const std::string backend = (be_env && be_env[0]) ? std::string(be_env)
                                                      : std::string("cpu");
    long action_steps = steps_env ? std::atol(steps_env) : 50;
    long action_dim   = dim_env   ? std::atol(dim_env)   : 32;
    if (action_steps <= 0 || action_dim <= 0) {
        std::printf("SKIP - bad FLASHRT_PI0_ACTION_STEPS/DIM\n");
        return 0;
    }

    int iw = 0, ih = 0, ic = 0;
    unsigned char * img = stbi_load(img_path.c_str(), &iw, &ih, &ic, 3);
    CHECK(img != nullptr && iw == 224 && ih == 224, "load image.png 224x224");
    int ww = 0, wh = 0, wc = 0;
    unsigned char * wrist = stbi_load(wrist_path.c_str(), &ww, &wh, &wc, 3);
    CHECK(wrist != nullptr && ww == 224 && wh == 224, "load wrist_image.png 224x224");
    bool prompt_ok = false;
    std::string prompt = read_file(prompt_path, &prompt_ok);
    CHECK(prompt_ok && !prompt.empty(), "prompt.txt non-empty");
    if (!prompt.empty() && prompt.back() == '\n') prompt.pop_back();

    if (!img || !wrist || !prompt_ok) {
        if (img)   stbi_image_free(img);
        if (wrist) stbi_image_free(wrist);
        std::printf(g_fail ? "\n== PI0 STATE PARITY FAILED ==\n"
                           : "\n== PI0 STATE PARITY SKIPPED ==\n");
        return g_fail;
    }

    const size_t n_elems = static_cast<size_t>(action_steps) * action_dim;
    const size_t n_bytes = n_elems * sizeof(float);
    const long state_dim = action_dim;  // backend exposes state on action_dim today

    const frt_llama_cpp_engine_factory_v1 * factory =
        frt_llama_cpp_default_engine_factory();
    CHECK(factory != nullptr, "default engine factory is non-null");

    std::string json =
        std::string("{") +
        "\"model_family\":\"pi0\","
        "\"model_path\":\"" + model_env + "\","
        "\"mmproj_path\":\"" + mmproj_env + "\","
        "\"backend\":\"" + backend + "\","
        "\"n_views\":2,\"image_height\":224,\"image_width\":224,"
        "\"image_channels\":3,\"action_steps\":" + std::to_string(action_steps) +
        ",\"action_dim\":" + std::to_string(action_dim) + "}";

    auto run_with_state = [&](const std::vector<float> & state,
                              std::vector<float> & out) -> bool {
        frt_model_runtime_v1 * model = nullptr;
        if (frt_llama_cpp_pi0_runtime_open_with_engine_factory(
                json.c_str(), factory, &model) != 0 || !model) {
            std::printf("FAIL: open FlashRT Pi0 runtime: %s\n",
                        factory->last_error(factory->self));
            g_fail = 1;
            return false;
        }
        frt_image_view views[2];
        views[0].struct_size = sizeof(frt_image_view);
        views[0].pixel_format = FRT_RT_PIXEL_RGB8;
        views[0].data = img;
        views[0].bytes = static_cast<uint64_t>(224) * 224 * 3;
        views[0].width = 224; views[0].height = 224; views[0].stride_bytes = 0;
        views[0].reserved = 0; views[0].timestamp_ns = 0;
        views[1] = views[0];
        views[1].data = wrist;
        bool ok = true;
        ok &= model->verbs.set_input(model->self,
            FRT_LLAMA_CPP_PI0_PORT_IMAGES, &views[0], sizeof(views), -1) == 0;
        ok &= model->verbs.set_input(model->self,
            FRT_LLAMA_CPP_PI0_PORT_PROMPT, prompt.data(), prompt.size(), -1) == 0;
        ok &= model->verbs.set_input(model->self,
            FRT_LLAMA_CPP_PI0_PORT_STATE, state.data(), state.size() * sizeof(float),
            -1) == 0;
        ok &= model->verbs.step(model->self) == 0;
        uint64_t written = 0;
        ok &= model->verbs.get_output(model->self,
            FRT_LLAMA_CPP_PI0_PORT_ACTIONS, out.data(), n_bytes, &written, -1) == 0;
        ok &= written == n_bytes;
        model->release(model->owner);
        return ok;
    };

    // Baseline: zero state (all bins map to the -1..0 region).
    std::vector<float> zero_state(state_dim, 0.0f);
    std::vector<float> actions_zero(n_elems, 0.0f);
    CHECK(run_with_state(zero_state, actions_zero), "infer with zero state");

    // Boundary-region states exercising every openpi bin branch. Each must
    // differ from the zero-state baseline once PI0.5 state is serialized into
    // the prompt. States are pre-normalized to [-1,1] (caller contract).
    struct Case { const char * name; float value; };
    const Case cases[] = {
        {"x<-1 (literal -1 token)", -2.0f},
        {"x in (-1,0)",              -0.5f},
        {"x in (0,1)",                0.5f},
        {"x>=1 (bin 255)",            1.5f},
    };
    for (const Case & c : cases) {
        std::vector<float> s(state_dim, c.value);
        std::vector<float> a(n_elems, 0.0f);
        CHECK(run_with_state(s, a), std::string("infer with ") + c.name);
        float max_diff = 0.0f;
        for (size_t i = 0; i < n_elems; ++i) {
            max_diff = std::max(max_diff, std::fabs(a[i] - actions_zero[i]));
        }
        std::printf("    %s: max abs diff vs zero-state = %.9g\n", c.name, max_diff);
        // Until the backend lands, this diff is 0 (state is a no-op for PI0.5),
        // so the check is intentionally NOT asserted here — it activates when
        // FLASHRT_PI0_STATE_PARITY_READY=1 implies the backend is fixed.
        CHECK(max_diff > 0.0f,
              std::string("state in prompt changes actions: ") + c.name);
    }

    if (img)   stbi_image_free(img);
    if (wrist) stbi_image_free(wrist);

    std::printf(g_fail ? "\n== PI0 STATE PARITY FAILED ==\n"
                       : "\n== PI0 STATE PARITY PASSED ==\n");
    return g_fail;
}
