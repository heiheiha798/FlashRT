/* test_model_runtime.cpp — unit acceptance for the model-runtime ABI.
 *
 * GPU-free on purpose: the builder and the wrap path only record opaque
 * handles and strings — nothing is dereferenced — so the ABI mechanics
 * (identity, fingerprint, lifetime, validation, verb plumbing) are testable
 * with fabricated handles. Execution over real graphs is covered by the
 * model-level tests (cpp/tests) and the consumer-side suites.
 */
#include "flashrt/model_runtime.h"

#include <cstdio>
#include <cstring>
#include <string>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

/* fabricated opaque handles — never dereferenced by this layer */
static frt_ctx    FAKE_CTX = (frt_ctx)0x10;
static frt_graph  FAKE_G0  = (frt_graph)0x20;
static frt_graph  FAKE_G1  = (frt_graph)0x21;
static frt_buffer FAKE_B0  = (frt_buffer)0x30;

struct VerbLog {
    int set_input = 0, get_output = 0, prepare = 0, step = 0, run_stage = 0;
    uint32_t last_stage = UINT32_MAX;
    int last_stream = -99;
};
static int v_set_input(void* s, uint32_t, const void*, uint64_t, int) {
    ((VerbLog*)s)->set_input++; return 0;
}
static int v_get_output(void* s, uint32_t, void*, uint64_t, uint64_t*, int) {
    ((VerbLog*)s)->get_output++; return 0;
}
static int v_prepare(void* s, uint32_t, frt_shape_key) {
    ((VerbLog*)s)->prepare++; return 0;
}
static int v_step(void* s) { ((VerbLog*)s)->step++; return 0; }
static int v_run_stage(void* s, uint32_t stage, int stream) {
    auto* log = (VerbLog*)s;
    log->run_stage++;
    log->last_stage = stage;
    log->last_stream = stream;
    return 0;
}
static const char* v_last_error(void*) { return ""; }

struct Owner { int retains = 0, releases = 0; };
static void owner_retain(void* p)  { ((Owner*)p)->retains++; }
static void owner_release(void* p) { ((Owner*)p)->releases++; }

static frt_runtime_builder make_builder() {
    frt_runtime_builder b = frt_runtime_builder_create(FAKE_CTX);
    frt_runtime_builder_add_stream(b, "main", 0, 0, nullptr);
    frt_shape_key keys[1] = {0};
    frt_runtime_builder_add_graph(b, "encode", FAKE_G0, 0, keys, 1, 0);
    frt_runtime_builder_add_graph(b, "decode", FAKE_G1, 0, keys, 1, 0);
    frt_runtime_builder_add_buffer(b, "b0", FAKE_B0, 4096, 1);
    frt_runtime_builder_add_identity(b, "model", "unit");
    return b;
}

static const int64_t IMG_SHAPE[4] = {3, 224, 224, 3};
static const int64_t ACT_SHAPE[2] = {50, 32};

static void add_ports_and_stages(frt_runtime_builder b, int64_t img_h = 224,
                                 uint64_t act_bytes = 3200) {
    const int64_t img[4] = {3, img_h, 224, 3};
    frt_runtime_builder_add_port(b, "images", FRT_RT_MOD_IMAGE,
                                 FRT_RT_DTYPE_BF16, FRT_RT_LAYOUT_NHWC,
                                 FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1,
                                 img, 4, 30, nullptr, 0, 0);
    frt_runtime_builder_add_port(b, "actions", FRT_RT_MOD_ACTION,
                                 FRT_RT_DTYPE_BF16, FRT_RT_LAYOUT_FLAT,
                                 FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0,
                                 ACT_SHAPE, 2, 0, FAKE_B0, 0, act_bytes);
    const uint32_t after1[1] = {0};
    frt_runtime_builder_add_stage(b, 0, nullptr, 0);
    frt_runtime_builder_add_stage(b, 1, after1, 1);
}

int main() {
    /* --- validation --- */
    {
        frt_runtime_builder b = make_builder();
        CHECK(frt_runtime_builder_add_stage(b, 7, nullptr, 0) < 0,
              "add_stage rejects out-of-range graph index");
        const uint32_t bad[1] = {0};
        CHECK(frt_runtime_builder_add_stage(b, 0, bad, 1) < 0,
              "add_stage rejects a dep on a not-yet-declared stage");
        CHECK(frt_runtime_builder_add_port(b, "", FRT_RT_MOD_TENSOR, 0, 0,
                                           FRT_RT_PORT_IN, FRT_RT_PORT_SWAP, 0,
                                           nullptr, 0, 0, nullptr, 0, 0) < 0,
              "add_port rejects an empty name");
        add_ports_and_stages(b);
        CHECK(frt_runtime_builder_finish(b, nullptr, nullptr, nullptr) == nullptr,
              "plain finish refuses a builder that declared ports/stages");
        /* the builder survives that refusal — finish_model consumes it */
        frt_model_runtime_v1* m = frt_runtime_builder_finish_model(
            b, nullptr, nullptr, nullptr, nullptr, nullptr);
        CHECK(m != nullptr, "finish_model after refused finish");
        /* absent producer verbs become unsupported stubs, never null */
        CHECK(m->verbs.set_input && m->verbs.step && m->verbs.last_error,
              "null verbs are stubbed");
        CHECK(m->verbs.set_input(m->self, 0, nullptr, 0, -1) == -3 &&
                  m->verbs.step(m->self) == -3 &&
                  m->verbs.last_error(m->self)[0] != '\0',
              "stubs report unsupported (-3) with an explanation");
        m->release(m->owner);
    }

    /* --- integrated build: struct, identity, fingerprint, verbs --- */
    Owner owner;
    VerbLog vlog;
    frt_model_runtime_verbs verbs{};
    verbs.struct_size = sizeof(verbs);
    verbs.set_input = v_set_input;
    verbs.get_output = v_get_output;
    verbs.prepare = v_prepare;
    verbs.step = v_step;
    verbs.last_error = v_last_error;

    frt_runtime_builder b = make_builder();
    add_ports_and_stages(b);
    frt_model_runtime_v1* m = frt_runtime_builder_finish_model(
        b, &verbs, &vlog, &owner, owner_retain, owner_release);
    CHECK(m != nullptr, "finish_model");
    CHECK(m->abi_version == FRT_MODEL_RUNTIME_ABI_VERSION &&
          m->struct_size == sizeof(frt_model_runtime_v1), "model ABI stamp");
    CHECK(m->exp && m->exp->abi_version == FRT_RUNTIME_ABI_VERSION,
          "embedded export is stamped");
    CHECK(m->n_ports == 2 && m->n_stages == 2, "port/stage counts");
    CHECK(std::strcmp(m->ports[0].name, "images") == 0 &&
          m->ports[0].update == FRT_RT_PORT_STAGED &&
          m->ports[0].rank == 4 && m->ports[0].shape[1] == 224,
          "port desc round-trips");
    CHECK(m->stages[1].graph == 1 && m->stages[1].n_after == 1 &&
          m->stages[1].after[0] == 0, "stage DAG round-trips");

    std::string id = m->exp->identity;
    CHECK(id.find("port:0:images:1:3:2:0:1:1:3,224,224,3:-1:0:0") !=
              std::string::npos,
          "identity carries the port schema (staged-only window = -1)");
    CHECK(id.find("port:1:actions:4:3:0:1:1:0:50,32:0:0:3200") !=
              std::string::npos,
          "identity carries the bound window (buffer index/offset/bytes)");
    CHECK(id.find("stage:1:1:0") != std::string::npos,
          "identity carries the stage DAG");
    CHECK(m->exp->fingerprint ==
              frt_runtime_fingerprint(id.data(), id.size()),
          "fingerprint == hash(identity)");

    /* port schema change => different fingerprint */
    {
        frt_runtime_builder b2 = make_builder();
        add_ports_and_stages(b2, /*img_h=*/256);
        frt_model_runtime_v1* m2 = frt_runtime_builder_finish_model(
            b2, &verbs, &vlog, nullptr, nullptr, nullptr);
        CHECK(m2 && m2->exp->fingerprint != m->exp->fingerprint,
              "port shape change changes the fingerprint");
        m2->release(m2->owner);
    }
    /* bound-window change => different fingerprint (schema unchanged) */
    {
        frt_runtime_builder b3 = make_builder();
        add_ports_and_stages(b3, /*img_h=*/224, /*act_bytes=*/6400);
        frt_model_runtime_v1* m3 = frt_runtime_builder_finish_model(
            b3, &verbs, &vlog, nullptr, nullptr, nullptr);
        CHECK(m3 && m3->exp->fingerprint != m->exp->fingerprint,
              "port window change changes the fingerprint");
        m3->release(m3->owner);
    }
    /* graph stream placement is deployment identity too */
    {
        frt_runtime_builder b4 = frt_runtime_builder_create(FAKE_CTX);
        frt_runtime_builder_add_stream(b4, "main", 0, 0, nullptr);
        frt_runtime_builder_add_stream(b4, "aux", 1, -1, nullptr);
        frt_shape_key keys[1] = {0};
        frt_runtime_builder_add_graph(b4, "encode", FAKE_G0, 0, keys, 1, 1);
        frt_runtime_builder_add_graph(b4, "decode", FAKE_G1, 0, keys, 1, 1);
        frt_runtime_builder_add_buffer(b4, "b0", FAKE_B0, 4096, 1);
        frt_runtime_builder_add_identity(b4, "model", "unit");
        add_ports_and_stages(b4);
        frt_model_runtime_v1* m4 = frt_runtime_builder_finish_model(
            b4, &verbs, &vlog, nullptr, nullptr, nullptr);
        CHECK(m4 && m4->exp->fingerprint != m->exp->fingerprint,
              "graph stream change changes the fingerprint");
        m4->release(m4->owner);
    }

    /* verbs plumb through self */
    m->verbs.set_input(m->self, 0, nullptr, 0, -1);
    m->verbs.get_output(m->self, 1, nullptr, 0, nullptr, -1);
    m->verbs.prepare(m->self, 0, 0);
    m->verbs.step(m->self);
    CHECK(vlog.set_input == 1 && vlog.get_output == 1 && vlog.prepare == 1 &&
          vlog.step == 1, "verbs dispatch through self");

    /* one refcount for the whole object: consumer retain via the model keeps
     * the embedded export alive too */
    CHECK(owner.retains == 1, "finish_model retained the owner once");
    m->retain(m->owner);
    m->release(m->owner);
    CHECK(owner.releases == 0, "still referenced after paired retain/release");
    m->release(m->owner);
    CHECK(owner.releases == 1, "final release frees the owner exactly once");

    /* --- adapter path: wrap an existing export --- */
    {
        Owner eo;                        /* export owner counters   */
        int wrapper_freed = 0;
        frt_runtime_builder eb = make_builder();
        frt_runtime_export_v1* exp = frt_runtime_builder_finish(
            eb, &eo, owner_retain, owner_release);
        CHECK(exp != nullptr, "plain export for the wrap path");

        frt_runtime_port_desc ports[1] = {};
        ports[0].name = "images";
        ports[0].modality = FRT_RT_MOD_IMAGE;
        ports[0].dtype = FRT_RT_DTYPE_BF16;
        ports[0].layout = FRT_RT_LAYOUT_NHWC;
        ports[0].direction = FRT_RT_PORT_IN;
        ports[0].update = FRT_RT_PORT_STAGED;
        ports[0].required = 1;
        ports[0].shape = IMG_SHAPE;
        ports[0].rank = 4;
        frt_runtime_stage_desc stages[1] = {};
        stages[0].graph = 0;

        frt_runtime_stage_desc bad = {};
        bad.graph = 9;
        CHECK(frt_model_runtime_wrap(exp, ports, 1, &bad, 1, &verbs, &vlog,
                                     nullptr, nullptr) == nullptr,
              "wrap rejects a stage over a missing graph");

        frt_model_runtime_v1* wm = frt_model_runtime_wrap(
            exp, ports, 1, stages, 1, &verbs, &vlog, &wrapper_freed,
            [](void* p) { *(int*)p += 1; });
        CHECK(wm != nullptr, "frt_model_runtime_wrap");
        CHECK(wm->exp == exp, "wrap keeps the export pointer");
        CHECK(std::strcmp(wm->ports[0].name, "images") == 0 &&
              wm->ports[0].shape != IMG_SHAPE,
              "wrap copies descriptor storage");
        /* wrapper took one export ref; drop the producer's */
        exp->release(exp->owner);
        CHECK(eo.releases == 0, "export alive while the wrapper holds it");
        wm->release(wm->owner);
        CHECK(wrapper_freed == 1 && eo.releases == 1,
              "wrapper release frees the producer instance and the export");
    }

    /* --- verb override path: inherit declarations, replace only verbs --- */
    {
        Owner base_owner;
        VerbLog base_vlog;
        frt_runtime_builder ob = make_builder();
        add_ports_and_stages(ob);
        frt_model_runtime_v1* base = frt_runtime_builder_finish_model(
            ob, &verbs, &base_vlog, &base_owner, owner_retain,
            owner_release);
        CHECK(base != nullptr, "override base model");

        Owner native_owner;
        VerbLog native_vlog;
        frt_model_runtime_verbs native_verbs{};
        native_verbs.struct_size = sizeof(native_verbs);
        native_verbs.set_input = v_set_input;
        native_verbs.get_output = v_get_output;
        native_verbs.prepare = v_prepare;
        native_verbs.step = v_step;
        native_verbs.last_error = v_last_error;

        frt_model_runtime_v1* over = frt_model_runtime_override_verbs(
            base, &native_verbs, &native_vlog, &native_owner, owner_retain,
            owner_release);
        CHECK(over != nullptr, "frt_model_runtime_override_verbs");
        CHECK(over->exp == base->exp && over->ports == base->ports &&
                  over->stages == base->stages,
              "override inherits export, ports, and stages by reference");
        CHECK(over->n_ports == 2 && over->n_stages == 2 &&
                  std::strcmp(over->ports[0].name, "images") == 0 &&
                  over->stages[1].after[0] == 0,
              "override preserves producer declarations");
        CHECK(over->exp->fingerprint == base->exp->fingerprint,
              "override does not change deployment identity");

        over->verbs.set_input(over->self, 0, nullptr, 0, -1);
        over->verbs.get_output(over->self, 1, nullptr, 0, nullptr, -1);
        over->verbs.prepare(over->self, 0, 0);
        over->verbs.step(over->self);
        CHECK(native_vlog.set_input == 1 && native_vlog.get_output == 1 &&
                  native_vlog.prepare == 1 && native_vlog.step == 1 &&
                  base_vlog.step == 0,
              "override verbs dispatch to the native object only");

        CHECK(base_owner.retains == 1 && native_owner.retains == 1,
              "override retains the model handle and native verb owner");
        base->release(base->owner);
        CHECK(base_owner.releases == 0,
              "base model stays alive while override references it");
        over->release(over->owner);
        CHECK(native_owner.releases == 1 && base_owner.releases == 1,
              "override release frees native owner and drops the base model");
    }

    /* --- v2 provider-owned callback runtime ----------------------------- */
    {
        Owner provider_owner;
        VerbLog provider_vlog;
        frt_runtime_builder pb = frt_runtime_builder_create_provider_owned();

        const int64_t text_shape[1] = {-1};
        const int64_t action_shape[2] = {50, 32};
        CHECK(frt_runtime_builder_add_port(
                  pb, "prompt", FRT_RT_MOD_TEXT, FRT_RT_DTYPE_U8,
                  FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_STAGED,
                  1, text_shape, 1, 0, nullptr, 0, 0) == 0,
              "provider-owned v2 accepts staged text input");
        CHECK(frt_runtime_builder_add_port(
                  pb, "actions", FRT_RT_MOD_ACTION, FRT_RT_DTYPE_F32,
                  FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED,
                  0, action_shape, 2, 0, nullptr, 0, 0) == 0,
              "provider-owned v2 accepts staged action output");
        CHECK(frt_runtime_builder_add_callback_stage_v2(
                  pb, "infer", 7, nullptr, 0) == 0,
              "provider-owned v2 accepts callback stage");

        frt_model_runtime_verbs_v2 verbs2{};
        verbs2.struct_size = sizeof(verbs2);
        verbs2.set_input = v_set_input;
        verbs2.get_output = v_get_output;
        verbs2.prepare = v_prepare;
        verbs2.step = v_step;
        verbs2.last_error = v_last_error;
        verbs2.run_stage = v_run_stage;

        frt_model_runtime_v2* mv2 = frt_runtime_builder_finish_model_v2(
            pb, &verbs2, &provider_vlog, &provider_owner, owner_retain,
            owner_release);
        CHECK(mv2 != nullptr, "finish provider-owned model_runtime_v2");
        CHECK(mv2->abi_version == FRT_MODEL_RUNTIME_ABI_VERSION_V2 &&
                  mv2->struct_size == sizeof(frt_model_runtime_v2),
              "model runtime v2 ABI stamp");
        CHECK(mv2->exp && mv2->exp->ctx == nullptr &&
                  mv2->exp->n_graphs == 0,
              "provider-owned v2 export has no FlashRT exec graph");
        CHECK(mv2->n_stages == 0 && mv2->n_stages_v2 == 1,
              "callback stage is visible only in the v2 stage view");
        CHECK(std::strcmp(mv2->stages_v2[0].name, "infer") == 0 &&
                  mv2->stages_v2[0].kind == FRT_RT_STAGE_CALLBACK &&
                  mv2->stages_v2[0].callback == 7,
              "callback stage descriptor round-trips");
        std::string id2 = mv2->exp->identity;
        CHECK(id2.find("stage_v2:0:infer:1:4294967295:7:") !=
                  std::string::npos,
              "identity carries v2 callback stage");

        mv2->verbs_v2.run_stage(mv2->self, 0, -1);
        mv2->verbs_v2.set_input(mv2->self, 0, nullptr, 0, -1);
        CHECK(provider_vlog.run_stage == 1 &&
                  provider_vlog.last_stage == 0 &&
                  provider_vlog.last_stream == -1 &&
                  provider_vlog.set_input == 1,
              "v2 verbs dispatch through self");
        CHECK(provider_owner.retains == 1, "v2 finish retained owner once");
        mv2->release(mv2->owner);
        CHECK(provider_owner.releases == 1,
              "v2 final release frees owner exactly once");
    }

    {
        frt_runtime_builder pb = frt_runtime_builder_create_provider_owned();
        CHECK(frt_runtime_builder_finish_model(pb, nullptr, nullptr, nullptr,
                                               nullptr, nullptr) == nullptr,
              "provider-owned builders cannot finish as v1 runtimes");
    }

    {
        Owner mixed_owner;
        VerbLog mixed_vlog;
        frt_runtime_builder mb = make_builder();
        const int64_t scalar_shape[1] = {1};
        CHECK(frt_runtime_builder_add_port(
                  mb, "input", FRT_RT_MOD_TENSOR, FRT_RT_DTYPE_F32,
                  FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_STAGED,
                  1, scalar_shape, 1, 0, nullptr, 0, 0) == 0,
              "mixed v2 test port");
        const uint32_t after0[1] = {0};
        CHECK(frt_runtime_builder_add_graph_stage_v2(
                  mb, "prefill", 0, nullptr, 0) == 0,
              "mixed v2 accepts graph stage");
        CHECK(frt_runtime_builder_add_callback_stage_v2(
                  mb, "decode", 3, after0, 1) == 0,
              "mixed v2 accepts callback stage after graph stage");
        frt_model_runtime_verbs_v2 verbs2{};
        verbs2.struct_size = sizeof(verbs2);
        verbs2.step = v_step;
        verbs2.last_error = v_last_error;
        verbs2.run_stage = v_run_stage;
        frt_model_runtime_v2* mv2 = frt_runtime_builder_finish_model_v2(
            mb, &verbs2, &mixed_vlog, &mixed_owner, owner_retain,
            owner_release);
        CHECK(mv2 && mv2->n_stages == 0 && mv2->n_stages_v2 == 2 &&
                  mv2->stages_v2[1].after[0] == 0,
              "mixed callback DAG has no legacy graph-only stage view");
        mv2->release(mv2->owner);
        CHECK(mixed_owner.releases == 1, "mixed v2 release frees owner");
    }

    std::printf(g_fail ? "\n== MODEL RUNTIME ABI FAILED ==\n"
                       : "\n== MODEL RUNTIME ABI PASSED ==\n");
    return g_fail;
}
