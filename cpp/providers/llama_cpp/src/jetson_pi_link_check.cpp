#include "flashrt/providers/llama_cpp/c_api.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstring>

extern "C" int frt_llama_cpp_jetson_pi_link_check(void) {
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    llama_model_params model_params = llama_model_default_params();
    llama_context_params context_params = llama_context_default_params();
    (void)mtmd_params;
    (void)model_params;
    (void)context_params;

    const char* marker = mtmd_default_marker();
    if (!marker || std::strlen(marker) == 0) return -1;

    /* Force a link-time reference to the Phase 1 Pi0 whole-graph entry point
     * (CLAUDE.md: mtmd_helper_eval_chunks_pi0 is the priority provider entry).
     * The default-param symbols above do not by themselves exercise this TU's
     * dependence on the Pi0 infer path. The `volatile` store defeats DCE so
     * the address must be materialized as an undefined symbol resolved from
     * libmtmd at link time. */
    volatile auto eval_pi0 = &mtmd_helper_eval_chunks_pi0;
    (void)eval_pi0;

    frt_llama_cpp_pi0_config cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.model_path = "model.gguf";
    cfg.mmproj_path = "mmproj.gguf";
    cfg.backend = "cpu";
    cfg.n_views = 1;
    cfg.image_height = 224;
    cfg.image_width = 224;
    cfg.image_channels = 3;
    cfg.state_dim = 8;
    cfg.action_steps = 2;
    cfg.action_dim = 4;

    frt_llama_cpp_engine_v1 engine{};
    engine.struct_size = sizeof(engine);
    frt_model_runtime_v2* model = nullptr;
    if (frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &engine, &model) !=
        -1) {
        if (model) model->release(model->owner);
        return -1;
    }
    return 0;
}
