// Numerical-parity test: FlashRT's Jetson-PI Pi0 provider (via
// frt_model_runtime_v1) vs a DIRECT jetson_pi_pi0 call, same inputs / backend
// / thread count. Proves the FlashRT port/stage/verb plumbing does not
// perturb the action chunk relative to the native narrow C API.
//
// Closes jetsonpi迁移.txt §14 (parity vs native). Skips (returns 0) when the
// weights/fixture env vars are unset, so CI without weights still passes.
//
// Determinism foundation: Pi0's action-chunk noise latent is
// create_normal_noise_cpp11(..., seed=41) (Jetson-PI/src/llama-context.cpp),
// reset to fresh noise + action_ready=false per tick; KV/position are cleared
// before every infer. So given fixed model+inputs+backend+n_threads, both
// paths must produce identical actions (modulo float-arith ULP). The existing
// tick-repro test already relies on this (test_llama_cpp_jetson_pi_engine.cpp).
//
// FlashRT adds NO numerical perturbation vs the direct call (verified): image
// swizzle is an identity copy for RGB8 fixtures (load via stbi_load(...,3)),
// prompt/state are copied verbatim, marker injection + zero-pad-to-action_dim
// happen INSIDE jetson_pi_pi0_infer for both, backend->n_gpu_layers/use_gpu
// mapping is identical, n_threads=0 on both -> same hardware_concurrency().
// A parity failure means a real FlashRT-port bug.
//
// Env:
//   FLASHRT_PI0_MODEL        path to Pi0 policy GGUF
//   FLASHRT_PI0_MMPROJ       path to VIT mmproj GGUF
//   FLASHRT_PI0_FIXTURE_DIR  dir with image.png, wrist_image.png, state.bin, prompt.txt
//   FLASHRT_PI0_ACTION_STEPS (optional) override; default 50 (LIBERO base).
//                            pi0_base is 10 — set FLASHRT_PI0_ACTION_STEPS=10
//                            or the open fails with an action-shape mismatch.
//   FLASHRT_PI0_ACTION_DIM   (optional) override; default 32.
//   FLASHRT_PI0_BACKEND      (optional) "cpu" (default) or "cuda".
//                            CUDA may have ULP drift from atomics/warp reductions;
//                            the 1e-5 tolerance is headroom. CPU expects ~0.
//   FLASHRT_PI0_CLI_ACTION_LOG (optional) log emitted by llama-mtmd-cli for
//                            the same fixture and effective prompt. When set,
//                            the `Pi0 action: [...]` line must be bit-identical
//                            to FlashRT's action chunk.

#include "flashrt/providers/llama_cpp/c_api.h"
#include "llama_cpp_generic_plan.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"
#include "jetson_pi_pi0.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
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
    const std::string state_path  = fixture_dir + "/state.bin";
    const std::string prompt_path = fixture_dir + "/prompt.txt";
    if (!file_exists(img_path.c_str()) || !file_exists(wrist_path.c_str()) ||
        !file_exists(state_path.c_str()) || !file_exists(prompt_path.c_str())) {
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
    if (action_steps <= 0 || action_dim <= 0 || action_steps > 10000 ||
        action_dim > 10000) {
        std::printf("SKIP - bad FLASHRT_PI0_ACTION_STEPS/DIM\n");
        return 0;
    }

    // Load fixtures ONCE; both paths consume the same bytes.
    int iw = 0, ih = 0, ic = 0;
    unsigned char * img = stbi_load(img_path.c_str(), &iw, &ih, &ic, 3);
    CHECK(img != nullptr && iw == 224 && ih == 224, "load image.png 224x224");
    int ww = 0, wh = 0, wc = 0;
    unsigned char * wrist = stbi_load(wrist_path.c_str(), &ww, &wh, &wc, 3);
    CHECK(wrist != nullptr && ww == 224 && wh == 224, "load wrist_image.png 224x224");
    bool state_ok = false;
    std::string state_bytes = read_file(state_path, &state_ok);
    // State width is policy-specific and independent of action_dim: PI0.5
    // discretizes up to 8 proprioception values into the prompt (openpi width
    // 8), while legacy Pi0 consumes an action_dim tensor. Accept any nonzero
    // float-multiple fixture; the backend enforces the per-model bound
    // (n_state <= 8 for PI0.5, n_state <= action_dim for legacy Pi0).
    const size_t state_count =
        state_bytes.size() / sizeof(float);
    CHECK(state_ok && state_bytes.size() % sizeof(float) == 0 &&
              state_count > 0 && state_count <= 8,
          "state.bin is 1..8 float32 values (PI0.5 proprioception width)");
    bool prompt_ok = false;
    std::string prompt = read_file(prompt_path, &prompt_ok);
    CHECK(prompt_ok && !prompt.empty(), "prompt.txt non-empty");
    if (!prompt.empty() && prompt.back() == '\n') prompt.pop_back();

    if (!img || !wrist || !state_ok || !prompt_ok) {
        if (img)   stbi_image_free(img);
        if (wrist) stbi_image_free(wrist);
        std::printf(g_fail ? "\n== PI0 PARITY FAILED ==\n"
                           : "\n== PI0 PARITY SKIPPED ==\n");
        return g_fail;
    }

    const size_t n_elems = static_cast<size_t>(action_steps) * action_dim;
    const size_t n_bytes = n_elems * sizeof(float);

    // ---- PATH A: FlashRT frt_model_runtime_v1 wrapper ----------------------
    std::vector<float> actions_flashrt(n_elems, 0.0f);
    {
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
        frt_model_runtime_v1 * model = nullptr;
        int rc = frt_llama_cpp_pi0_runtime_open_with_engine_factory(
            json.c_str(), factory, &model);
        if (rc != 0 || !model) {
            std::printf("FAIL: open FlashRT Pi0 runtime (rc=%d): %s\n", rc,
                        factory->last_error(factory->self));
            g_fail = 1;
        } else {
            CHECK(true, "open FlashRT Pi0 runtime from JSON");
            const auto* plan = llama_cpp_generic_plan(model);
            // Pi0 publishes a single `infer` stage. The backend
            // jetson_pi_pi0_context()/action() pair is a cached result handoff,
            // not a real encode/decode boundary, so `context -> action` is not
            // advertised (see pi0_runtime.cpp and the linked Jetson-PI-Edge PR).
            // Re-enable the 2-stage plan once the backend exposes a genuine
            // pending-context/decode split and PI0.5 state parity is proven.
            CHECK(model->n_stages == 0 && plan && plan->n_stages == 1,
                  "FlashRT Pi0 runtime exposes a single infer stage");

            frt_image_view views[2];
            views[0].struct_size = sizeof(frt_image_view);
            views[0].pixel_format = FRT_RT_PIXEL_RGB8;
            views[0].data = img;
            views[0].bytes = static_cast<uint64_t>(224) * 224 * 3;
            views[0].width = 224; views[0].height = 224; views[0].stride_bytes = 0;
            views[0].reserved = 0; views[0].timestamp_ns = 0;
            views[1] = views[0];
            views[1].data = wrist;

            CHECK(model->verbs.set_input(model->self,
                                            FRT_LLAMA_CPP_PI0_PORT_IMAGES,
                                            &views[0], sizeof(views), -1) == 0,
                  "FlashRT set_input images");
            CHECK(model->verbs.set_input(model->self,
                                            FRT_LLAMA_CPP_PI0_PORT_PROMPT,
                                            prompt.data(), prompt.size(), -1) == 0,
                  "FlashRT set_input prompt");
            CHECK(model->verbs.set_input(model->self,
                                            FRT_LLAMA_CPP_PI0_PORT_STATE,
                                            state_bytes.data(),
                                            state_bytes.size(), -1) == 0,
                  "FlashRT set_input state");
            CHECK(model->verbs.step(model->self) == 0,
                  "FlashRT run_stage infer");

            uint64_t written = 0;
            rc = model->verbs.get_output(
                model->self, FRT_LLAMA_CPP_PI0_PORT_ACTIONS,
                actions_flashrt.data(), n_bytes, &written, -1);
            CHECK(rc == 0 && written == n_bytes,
                  "FlashRT get_output actions shape matches config");
            // No context/action split is exposed. Confirm the selected plan
            // does not advertise a "context" or "action" stage.
            CHECK(llama_cpp_run_generic_stage(model, "context") == -3,
                  "Pi0 provider does not advertise a context stage");
            CHECK(llama_cpp_run_generic_stage(model, "action") == -3,
                  "Pi0 provider does not advertise an action stage");
            model->release(model->owner);
        }
    }

    // ---- PATH B: direct jetson_pi_pi0 call ---------------------------------
    std::vector<float> actions_native(n_elems, 0.0f);
    {
        jetson_pi_pi0_config jc{};
        jc.struct_size   = sizeof(jc);
        jc.model_path    = model_env;
        jc.mmproj_path   = mmproj_env;
        jc.backend       = backend.c_str();
        jc.n_views       = 2;
        jc.image_height  = 224;
        jc.image_width   = 224;
        jc.n_threads     = 0;  // match FlashRT's hardcoded 0 (jetson_pi_engine.cpp)

        jetson_pi_pi0 * pi0 = nullptr;
        int32_t s = jetson_pi_pi0_open(&jc, &pi0);
        if (s != JETSON_PI_PI0_OK || !pi0) {
            std::printf("FAIL: jetson_pi_pi0_open (rc=%d): %s\n", s,
                        jetson_pi_pi0_open_error());
            g_fail = 1;
        } else {
            CHECK(true, "direct jetson_pi_pi0_open");

            uint32_t steps = 0, dim = 0;
            CHECK(jetson_pi_pi0_action_shape(pi0, &steps, &dim) ==
                      JETSON_PI_PI0_OK &&
                  steps == static_cast<uint32_t>(action_steps) &&
                  dim == static_cast<uint32_t>(action_dim),
                  "direct action_shape matches config");

            const uint8_t * imgs[2] = { img, wrist };
            size_t written = 0;
            CHECK(jetson_pi_pi0_action(
                      pi0, actions_native.data(), n_elems, &written) ==
                      JETSON_PI_PI0_ACTION_NOT_READY && written == 0,
                  "direct action-not-ready reports zero elements written");
            written = 0;
            CHECK(jetson_pi_pi0_infer(
                      pi0, imgs, 2, prompt.data(), prompt.size(),
                      reinterpret_cast<const float*>(state_bytes.data()),
                      state_bytes.size() / sizeof(float), actions_native.data(),
                      n_elems - 1, &written) ==
                      JETSON_PI_PI0_BUFFER_TOO_SMALL && written == n_elems,
                  "too-small whole infer reports required size before context");
            written = 123;
            CHECK(jetson_pi_pi0_action(
                      pi0, actions_native.data(), n_elems, &written) ==
                      JETSON_PI_PI0_ACTION_NOT_READY && written == 0,
                  "invalid whole infer leaves no consumable context");
            CHECK(jetson_pi_pi0_context(
                      pi0, imgs, 2, prompt.data(), prompt.size(),
                      reinterpret_cast<const float*>(state_bytes.data()),
                      state_bytes.size() / sizeof(float)) == JETSON_PI_PI0_OK,
                  "direct context prepares pending action");
            CHECK(jetson_pi_pi0_context(
                      pi0, imgs, 2, nullptr, 0,
                      reinterpret_cast<const float*>(state_bytes.data()),
                      state_bytes.size() / sizeof(float)) == JETSON_PI_PI0_INVALID,
                  "failed direct context replacement is rejected");
            written = 123;
            CHECK(jetson_pi_pi0_action(
                      pi0, actions_native.data(), n_elems, &written) ==
                      JETSON_PI_PI0_ACTION_NOT_READY && written == 0,
                  "failed direct context discards prior pending context");
            s = jetson_pi_pi0_infer(pi0, imgs, 2,
                                    prompt.data(), prompt.size(),
                                    reinterpret_cast<const float*>(state_bytes.data()),
                                    state_bytes.size() / sizeof(float),
                                    actions_native.data(), n_elems, &written);
            CHECK(s == JETSON_PI_PI0_OK && written == n_elems,
                  "direct jetson_pi_pi0_infer produced action chunk");
            if (s != JETSON_PI_PI0_OK) {
                std::printf("    infer error: %s\n", jetson_pi_pi0_last_error(pi0));
            }
            jetson_pi_pi0_close(pi0);
        }
    }

    if (img)   stbi_image_free(img);
    if (wrist) stbi_image_free(wrist);

    if (g_fail) {
        std::printf("\n== PI0 PARITY FAILED ==\n");
        return g_fail;
    }

    // ---- compare -----------------------------------------------------------
    bool nan_inf = false;
    for (float v : actions_flashrt) {
        if (std::isnan(v) || std::isinf(v)) { nan_inf = true; break; }
    }
    CHECK(!nan_inf, "FlashRT actions contain no NaN/Inf");

    // No context/action split is published, so there is no split path to
    // compare against the whole-infer path. The single `infer` stage output
    // is compared against the direct jetson_pi_pi0 call below.

    float max_diff = 0.0f;
    size_t diverge_idx = 0;
    for (size_t i = 0; i < n_elems; ++i) {
        float d = std::fabs(actions_flashrt[i] - actions_native[i]);
        if (d > max_diff) { max_diff = d; diverge_idx = i; }
    }
    std::printf("    max abs diff = %.3g  (n_elems=%zu, first diverge idx=%zu)\n",
                max_diff, n_elems, diverge_idx);
    CHECK(max_diff <= 1e-5f,
          "FlashRT actions match direct jetson_pi_pi0 (max_diff <= 1e-5)");

    const char * cli_action_log = std::getenv("FLASHRT_PI0_CLI_ACTION_LOG");
    if (cli_action_log && cli_action_log[0]) {
        std::ifstream log(cli_action_log);
        CHECK(log.good(), "open native Pi0 CLI action log");
        std::vector<float> actions_cli;
        std::string line;
        const std::string prefix = "Pi0 action: [";
        while (std::getline(log, line)) {
            if (line.rfind(prefix, 0) != 0 || line.back() != ']') {
                continue;
            }
            std::string payload = line.substr(
                prefix.size(), line.size() - prefix.size() - 1);
            for (char & ch : payload) {
                if (ch == ',') ch = ' ';
            }
            std::istringstream values(payload);
            float value = 0.0f;
            while (values >> value) actions_cli.push_back(value);
            if (!values.eof()) actions_cli.clear();
            break;
        }
        CHECK(actions_cli.size() == n_elems,
              "native Pi0 CLI log contains the complete action chunk");
        if (actions_cli.size() == n_elems) {
            float cli_max_diff = 0.0f;
            size_t cli_diverge_idx = 0;
            for (size_t i = 0; i < n_elems; ++i) {
                const float d = std::fabs(actions_flashrt[i] - actions_cli[i]);
                if (d > cli_max_diff) {
                    cli_max_diff = d;
                    cli_diverge_idx = i;
                }
            }
            std::printf("    CLI max abs diff = %.9g  (n_elems=%zu, first diverge idx=%zu)\n",
                        cli_max_diff, n_elems, cli_diverge_idx);
            CHECK(std::memcmp(actions_flashrt.data(), actions_cli.data(),
                              n_bytes) == 0,
                  "FlashRT actions are bit-identical to native Pi0 CLI");
        }
    }

    std::printf(g_fail ? "\n== PI0 PARITY FAILED ==\n"
                       : "\n== PI0 PARITY PASSED ==\n");
    return g_fail;
}
