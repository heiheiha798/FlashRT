"""Jetson-PI Pi0 frontend — drives the Jetson-PI llama.cpp/GGML Pi0 provider
through the FlashRT ``frt_model_runtime_v2`` C ABI via ctypes.

This frontend is the Phase 2 Python entry for the Jetson-PI provider. It
dlopens the SHARED provider library (``libflashrt_cpp_llama_cpp_provider_c.so``)
built under ``FLASHRT_CPP_WITH_JETSON_PI``, opens a Pi0 runtime through
``frt_llama_cpp_default_engine_factory`` + ``open_with_engine_factory``, and
drives one whole-graph Pi0 infer per ``infer(observation)`` call.

The frontend intentionally does no action unnormalization / LIBERO slicing:
it returns the raw ``action_steps × action_dim`` action chunk the model
produces. Higher layers (or the caller) post-process as needed.

Memory: the Jetson-PI engine copies all inputs on ``set_input`` (see
``jetson_pi_engine.cpp``), so numpy arrays need not be kept alive past the
``infer`` call.
"""

from __future__ import annotations

import ctypes
import json
import os

import numpy as np

# ---- frt_image_view ctypes mirror (matches runtime/include/flashrt/model_runtime.h) ----

FRT_RT_PIXEL_RGB8 = 0


class FrtImageView(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("bytes", ctypes.c_uint64),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("stride_bytes", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
    ]


# ---- frt_model_runtime_v2 verbs + struct mirrors (only the fields we touch) ----
#
# We only need: abi_version, struct_size (sanity), self, verbs_v2 (for
# set_input/get_output/run_stage/last_error), owner, release. The full struct
# has more fields (exp, ports, stages, verbs v1, retain, stages_v2) but ctypes
# only requires correct field ORDER up to the last one we read; trailing fields
# can be omitted as long as we never touch them. To stay robust against layout
# drift we mirror the full v2 layout.

_SetInputFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int)
_GetOutputFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_int)
_PrepareFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64)
_StepFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_LastErrorFn = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)
_RunStageFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int)
_RetainReleaseFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_TokenCopyToHostFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)
_TokenCopyFromHostFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)
_TokenSyncFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_TokenDestroyFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_TokenMapHostFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
    ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
_TokenUnmapHostFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


class _FrtMemoryTokenVerbs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("copy_to_host", _TokenCopyToHostFn),
        ("copy_from_host", _TokenCopyFromHostFn),
        ("sync", _TokenSyncFn),
        ("destroy", _TokenDestroyFn),
        ("map_host", _TokenMapHostFn),
        ("unmap_host", _TokenUnmapHostFn),
    ]


class _FrtMemoryTokenDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("handle", ctypes.c_void_p),
        ("verbs", ctypes.POINTER(_FrtMemoryTokenVerbs)),
        ("offset", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("location_kind", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _FrtVerbsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("set_input", _SetInputFn),
        ("get_output", _GetOutputFn),
        ("prepare", _PrepareFn),
        ("step", _StepFn),
        ("last_error", _LastErrorFn),
    ]


class _FrtVerbsV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("set_input", _SetInputFn),
        ("get_output", _GetOutputFn),
        ("prepare", _PrepareFn),
        ("step", _StepFn),
        ("last_error", _LastErrorFn),
        ("run_stage", _RunStageFn),
    ]


class FrtModelRuntimeV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("exp", ctypes.c_void_p),
        ("ports", ctypes.c_void_p),
        ("n_ports", ctypes.c_uint64),
        ("stages", ctypes.c_void_p),
        ("n_stages", ctypes.c_uint64),
        ("self", ctypes.c_void_p),
        ("verbs", _FrtVerbsV1),
        ("owner", ctypes.c_void_p),
        ("retain", _RetainReleaseFn),
        ("release", _RetainReleaseFn),
        ("stages_v2", ctypes.c_void_p),
        ("n_stages_v2", ctypes.c_uint64),
        ("verbs_v2", _FrtVerbsV2),
        ("port_tokens", ctypes.POINTER(_FrtMemoryTokenDesc)),
        ("n_port_tokens", ctypes.c_uint64),
    ]


class _MappedActions(np.ndarray):
    def __new__(cls, frontend, token, pointer):
        count = frontend.action_steps * frontend.action_dim
        buffer = (ctypes.c_float * count).from_address(pointer)
        view = np.ctypeslib.as_array(buffer).reshape(
            frontend.action_steps, frontend.action_dim).view(cls)
        view._frontend = frontend
        view._token = token
        view._pointer = pointer
        view._closed = False
        view._owns_mapping = True
        view._owner = frontend._model.owner
        view._release = frontend._model.release
        frontend._model.retain(view._owner)
        view.setflags(write=False)
        return view

    def __array_finalize__(self, source):
        if source is None:
            return
        self._frontend = getattr(source, "_frontend", None)
        self._token = getattr(source, "_token", None)
        self._pointer = getattr(source, "_pointer", None)
        self._closed = True
        self._owns_mapping = False
        self._owner = None
        self._release = None

    def close(self):
        if self._closed or not self._owns_mapping:
            return
        rc = self._token.verbs.contents.unmap_host(
            self._token.handle, self._pointer)
        if rc != 0:
            raise RuntimeError(f"unmap_host actions failed (rc={rc})")
        self._closed = True
        self._release(self._owner)

    def __dlpack__(self, *, stream=None, max_version=None,
                   dl_device=None, copy=None):
        if self._closed:
            raise RuntimeError("cannot export a closed actions host view")
        dlpack_view = self._frontend._map_actions()
        return np.ndarray.__dlpack__(
            dlpack_view, stream=stream, max_version=max_version,
            dl_device=dl_device, copy=copy)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Port indices (c_api.h: FRT_LLAMA_CPP_PI0_PORT_*)
PORT_IMAGES = 0
PORT_PROMPT = 1
PORT_STATE = 2
PORT_ACTIONS = 3
STAGE_INFER = 0
STAGE_CONTEXT = 1
STAGE_ACTION = 2

# frt_model_runtime_v2 abi_version (model_runtime.h: FRT_MODEL_RUNTIME_ABI_VERSION_V2)
_FRT_MODEL_RUNTIME_ABI_VERSION_V2 = 2
_F32_BYTES = np.dtype(np.float32).itemsize


def _find_lib(lib_path, env_var="FLASHRT_PI0_LIB"):
    """Resolve the SHARED provider .so path.

    Priority: explicit ``lib_path`` kwarg > ``env_var`` env > build-dir
    convention. An explicit ``lib_path`` that does not exist is a hard error
    (no silent fallback to env/build-dirs) so callers get deterministic loads.
    """
    if lib_path is not None:
        if not os.path.exists(lib_path):
            raise RuntimeError(
                f"lib_path does not exist: {lib_path}")
        return lib_path
    env = os.environ.get(env_var)
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    candidates = [
        os.path.join(repo, "cpp", "build-jetson-pi",
                     "libflashrt_cpp_llama_cpp_provider_c.so"),
        os.path.join(repo, "cpp", "build-container",
                     "libflashrt_cpp_llama_cpp_provider_c.so"),
        os.path.join(repo, "cpp", "build",
                     "libflashrt_cpp_llama_cpp_provider_c.so"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise RuntimeError(
        "libflashrt_cpp_llama_cpp_provider_c.so not found. Build it with "
        "-DFLASHRT_CPP_WITH_JETSON_PI=ON, or pass lib_path=, or set "
        f"{env_var}.")


# frt_llama_cpp_engine_factory_v1 layout:
#   uint32 struct_size; uint32 reserved; void* self;
#   int (*create_pi0)(void*, const cfg*, engine*);
#   int (*create_llm)(void*, const cfg*, engine*);   // Phase 3, between pi0 and last_error
#   const char* (*last_error)(void*);
# self is nullptr for the default factory; last_error ignores it (returns the
# thread-local create-error sink). Defined at module scope so it is not
# re-created per instance. Field order MUST match c_api.h exactly — ctypes
# reads by offset, so a missing field shifts every later field.
class _FactoryV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("self", ctypes.c_void_p),
        ("create_pi0", ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p)),
        ("create_llm", ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p)),
        ("create_mllm", ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p)),
        ("last_error", ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)),
    ]


class Pi0JetsonPiFrontend:
    """Pi0 VLA frontend backed by the Jetson-PI llama.cpp/GGML provider."""

    def __init__(self, checkpoint, *, mmproj_path=None, backend="cpu",
                 num_views=2, image_height=224, image_width=224,
                 action_steps=None, action_dim=None,
                 lib_path=None, **_unused):
        if mmproj_path is None:
            raise ValueError("mmproj_path is required for the Jetson-PI Pi0 frontend")
        if action_steps is None or action_dim is None:
            raise ValueError(
                "action_steps and action_dim must be set explicitly (e.g. 10x32 "
                "for pi0_base, 50x32 for pi0_libero_base)")
        self.num_views = int(num_views)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.action_steps = int(action_steps)
        self.action_dim = int(action_dim)
        self._prompt = b""
        self._lib_path = _find_lib(lib_path)
        self._lib = ctypes.CDLL(self._lib_path)

        # C ABI signatures.
        self._lib.frt_llama_cpp_default_engine_factory.argtypes = []
        self._lib.frt_llama_cpp_default_engine_factory.restype = ctypes.c_void_p

        # frt_llama_cpp_engine_factory_v1 { struct_size, reserved, self,
        #   create_pi0(self, config*, engine*) -> int, last_error(self) -> char* }
        # We only call create_pi0 and last_error through the returned pointer.
        self._lib.frt_llama_cpp_pi0_runtime_open_with_engine_factory.argtypes = [
            ctypes.c_char_p,           # config_json
            ctypes.c_void_p,           # factory*
            ctypes.POINTER(ctypes.c_void_p),  # out model**
        ]
        self._lib.frt_llama_cpp_pi0_runtime_open_with_engine_factory.restype = (
            ctypes.c_int)

        config = {
            "model_family": "pi0",
            "model_path": str(checkpoint),
            "mmproj_path": str(mmproj_path),
            "backend": backend,
            "n_views": int(num_views),
            "image_height": int(image_height),
            "image_width": int(image_width),
            "image_channels": 3,
            "action_steps": int(action_steps),
            "action_dim": int(action_dim),
        }
        # ensure_ascii=False: the C-side JSON parser (pi0_runtime.cpp) does
        # not handle \uXXXX escapes, only raw UTF-8 bytes.
        config_json = json.dumps(config, ensure_ascii=False).encode("utf-8")

        factory_ptr = self._lib.frt_llama_cpp_default_engine_factory()
        if not factory_ptr:
            raise RuntimeError(
                "frt_llama_cpp_default_engine_factory returned NULL "
                "(FlashRT built without FLASHRT_CPP_WITH_JETSON_PI?)")

        factory = _FactoryV1.from_address(factory_ptr)
        if factory.struct_size < ctypes.sizeof(_FactoryV1):
            raise RuntimeError(
                f"frt_llama_cpp_engine_factory_v1 struct_size="
                f"{factory.struct_size} < ctypes sizeof "
                f"{ctypes.sizeof(_FactoryV1)}; provider .so is older than the "
                f"Python frontend expects.")

        model_ptr = ctypes.c_void_p(0)
        rc = self._lib.frt_llama_cpp_pi0_runtime_open_with_engine_factory(
            config_json, factory_ptr, ctypes.byref(model_ptr))
        if rc != 0 or not model_ptr.value:
            err = factory.last_error(factory.self) or b""
            raise RuntimeError(
                f"frt_llama_cpp_pi0_runtime_open_with_engine_factory failed "
                f"(rc={rc}): {(err.decode(errors='replace') if err else 'no error')}")

        self._model = FrtModelRuntimeV2.from_address(model_ptr.value)
        # ABI gate: refuse to drive a struct laid out for a different ABI
        # version (mirrors the check in runtime/bindings/runtime_pybind.cpp).
        if self._model.abi_version != _FRT_MODEL_RUNTIME_ABI_VERSION_V2:
            raise RuntimeError(
                f"frt_model_runtime_v2 abi_version={self._model.abi_version}, "
                f"expected {_FRT_MODEL_RUNTIME_ABI_VERSION_V2}; the provider "
                f".so was built against a different FlashRT runtime ABI.")
        if self._model.struct_size < ctypes.sizeof(FrtModelRuntimeV2):
            raise RuntimeError(
                f"frt_model_runtime_v2 struct_size={self._model.struct_size} "
                f"< ctypes sizeof {ctypes.sizeof(FrtModelRuntimeV2)}; the "
                f"provider .so is older than the Python frontend expects.")
        self._model_ptr = model_ptr  # keep the uintptr for sanity

    # -- VLAModel.predict contract -------------------------------------------

    def set_prompt(self, prompt_text):
        # Note: this signature does NOT accept `state` — VLAModel.predict
        # detects that and routes state through observation["state"] instead.
        if isinstance(prompt_text, bytes):
            self._prompt = prompt_text
        else:
            self._prompt = (prompt_text or "").encode("utf-8")

    def infer(self, observation, debug=False):
        if self._model is None:
            raise RuntimeError("Pi0JetsonPiFrontend is closed")
        _ = debug  # accepted for VLAModel.predict signature parity; unused
        # Collect images: predict() passes obs with 'images' list + 'image'/
        # 'wrist_image' legacy keys. Prefer the explicit list.
        if "images" in observation:
            images = list(observation["images"])
        else:
            images = [observation["image"]]
            if "wrist_image" in observation:
                images.append(observation["wrist_image"])
        if len(images) != self.num_views:
            raise ValueError(
                f"expected {self.num_views} images, got {len(images)}")
        views = self._make_image_views(images)

        state = observation.get("state")
        if state is None:
            raise ValueError("observation['state'] is required for the Jetson-PI Pi0 frontend")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size > self.action_dim:
            raise ValueError(
                f"state has {state.size} values, more than action_dim={self.action_dim}")
        state_padded = np.zeros(self.action_dim, dtype=np.float32)
        state_padded[:state.size] = state

        v2 = self._model.verbs_v2
        self_ = self._model.self

        rc = v2.set_input(self_, PORT_IMAGES, ctypes.cast(views, ctypes.c_void_p),
                          ctypes.sizeof(FrtImageView) * len(images), -1)
        self._check(rc, "set_input images")
        rc = v2.set_input(self_, PORT_PROMPT, self._prompt,
                          len(self._prompt), -1)
        self._check(rc, "set_input prompt")
        rc = v2.set_input(self_, PORT_STATE, state_padded.ctypes.data,
                          state_padded.nbytes, -1)
        self._check(rc, "set_input state")

        rc = v2.run_stage(self_, STAGE_INFER, -1)
        self._check(rc, "run_stage infer")

        capacity = self.action_steps * self.action_dim * _F32_BYTES
        out = (ctypes.c_char * capacity)()
        written = ctypes.c_uint64(0)
        rc = v2.get_output(self_, PORT_ACTIONS, out, capacity,
                           ctypes.byref(written), -1)
        self._check(rc, "get_output actions")
        need = capacity
        if written.value != need:
            raise RuntimeError(
                f"get_output wrote {written.value} bytes, expected {need}")
        actions = np.frombuffer(out, dtype=np.float32,
                                count=self.action_steps * self.action_dim
                                ).reshape(self.action_steps, self.action_dim).copy()
        return {"actions": actions}

    def context(self, observation):
        """Run the Pi0 context stage and retain provider-private encoded state."""
        if self._model is None:
            raise RuntimeError("Pi0JetsonPiFrontend is closed")
        images = list(observation["images"]) if "images" in observation else [
            observation["image"]] + (
                [observation["wrist_image"]]
                if "wrist_image" in observation else [])
        if len(images) != self.num_views:
            raise ValueError(
                f"expected {self.num_views} images, got {len(images)}")
        state = observation.get("state")
        if state is None:
            raise ValueError(
                "observation['state'] is required for the Jetson-PI Pi0 frontend")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size > self.action_dim:
            raise ValueError(
                f"state has {state.size} values, more than action_dim={self.action_dim}")
        state_padded = np.zeros(self.action_dim, dtype=np.float32)
        state_padded[:state.size] = state
        views = self._make_image_views(images)
        v2 = self._model.verbs_v2
        self_ = self._model.self
        self._check(v2.set_input(
            self_, PORT_IMAGES, ctypes.cast(views, ctypes.c_void_p),
            ctypes.sizeof(FrtImageView) * len(images), -1),
            "set_input images")
        self._check(v2.set_input(
            self_, PORT_PROMPT, self._prompt, len(self._prompt), -1),
            "set_input prompt")
        self._check(v2.set_input(
            self_, PORT_STATE, state_padded.ctypes.data,
            state_padded.nbytes, -1), "set_input state")
        self._check(v2.run_stage(self_, STAGE_CONTEXT, -1),
                    "run_stage context")

    def action(self):
        """Consume the pending Pi0 context and return one action chunk."""
        if self._model is None:
            raise RuntimeError("Pi0JetsonPiFrontend is closed")
        v2 = self._model.verbs_v2
        self._check(v2.run_stage(self._model.self, STAGE_ACTION, -1),
                    "run_stage action")
        capacity = self.action_steps * self.action_dim * _F32_BYTES
        out = (ctypes.c_char * capacity)()
        written = ctypes.c_uint64(0)
        self._check(v2.get_output(
            self._model.self, PORT_ACTIONS, out, capacity,
            ctypes.byref(written), -1), "get_output actions")
        if written.value != capacity:
            raise RuntimeError(
                f"get_output wrote {written.value} bytes, expected {capacity}")
        return np.frombuffer(
            out, dtype=np.float32,
            count=self.action_steps * self.action_dim).reshape(
                self.action_steps, self.action_dim).copy()

    def action_view(self):
        """Map the latest actions as a read-only zero-copy NumPy/DLPack view.

        The returned ndarray implements ``__dlpack__``. Close it before
        setting new inputs or running another stage; those operations reject
        while a host view is mapped so the backing pointer cannot go stale.
        """
        return self._map_actions()

    def _map_actions(self):
        if self._model is None:
            raise RuntimeError("Pi0JetsonPiFrontend is closed")
        if self._model.n_port_tokens != self._model.n_ports:
            raise RuntimeError("provider runtime has invalid port token layout")
        token = self._model.port_tokens[PORT_ACTIONS]
        if not token.handle or not token.verbs:
            raise RuntimeError("actions port does not expose a memory token")
        verbs = token.verbs.contents
        if verbs.struct_size < ctypes.sizeof(_FrtMemoryTokenVerbs):
            raise RuntimeError("actions token does not support host mapping")
        pointer = ctypes.c_void_p()
        rc = verbs.map_host(
            token.handle, token.offset, token.bytes, 1, ctypes.byref(pointer))
        if rc != 0 or not pointer.value:
            raise RuntimeError(f"map_host actions failed (rc={rc})")
        return _MappedActions(self, token, pointer.value)

    def close(self):
        if getattr(self, "_model", None) is not None:
            if self._model.release:
                self._model.release(self._model.owner)
            self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- helpers --------------------------------------------------------------

    def _make_image_views(self, images):
        views = (FrtImageView * len(images))()
        for i, im in enumerate(images):
            arr = np.ascontiguousarray(im, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(
                    f"image {i} must be HxWx3 uint8, got shape {arr.shape}")
            if arr.shape[0] != self.image_height or arr.shape[1] != self.image_width:
                raise ValueError(
                    f"image {i} must be {self.image_height}x{self.image_width}, "
                    f"got {arr.shape[0]}x{arr.shape[1]}")
            views[i].struct_size = ctypes.sizeof(FrtImageView)
            views[i].pixel_format = FRT_RT_PIXEL_RGB8
            views[i].data = ctypes.c_void_p(arr.ctypes.data)
            views[i].bytes = arr.nbytes
            views[i].width = int(arr.shape[1])
            views[i].height = int(arr.shape[0])
            views[i].stride_bytes = int(arr.strides[0])
            views[i].reserved = 0
            views[i].timestamp_ns = 0
            # Keep the array alive for the duration of the infer call by
            # stashing it on the views object (engine copies on set_input).
            setattr(views, f"_keepalive_{i}", arr)
        return views

    def _check(self, rc, what):
        if rc != 0:
            err = b""
            try:
                err = self._model.verbs_v2.last_error(self._model.self) or b""
            except Exception:
                pass
            raise RuntimeError(
                f"{what} failed (rc={rc}): "
                f"{(err.decode(errors='replace') if err else 'no error')}")
