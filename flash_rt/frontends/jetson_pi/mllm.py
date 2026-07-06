"""Jetson-PI multimodal LLM frontend — drives the Jetson-PI llama.cpp MLLM
provider through the FlashRT ``frt_model_runtime_v2`` C ABI via ctypes.

This is the Phase 4 Python entry for vision-language models (images + prompt →
text). The caller is responsible for applying the chat template; the engine
only does raw prompt + media markers → text.

``flash_rt.load_model(framework="jetson_pi", config="mllm", ...)`` returns a
:class:`MllmJetsonPiFrontend` directly (it is NOT wrapped in ``VLAModel`` —
MLLMs output text, not actions).
"""

from __future__ import annotations

import ctypes
import json
import os

import numpy as np

from .pi0 import (
    FrtImageView,
    FrtModelRuntimeV2,
    FRT_RT_PIXEL_RGB8,
    _FactoryV1,
    _FRT_MODEL_RUNTIME_ABI_VERSION_V2,
    _find_lib,
)

PORT_IMAGES = 0
PORT_PROMPT = 1
PORT_TEXT = 2
STAGE_INFER = 0


class MllmJetsonPiFrontend:
    """Multimodal LLM frontend backed by the Jetson-PI provider."""

    def __init__(self, checkpoint, *, mmproj_path, backend="cpu",
                 n_ctx=0, n_threads=0, temp=0.8, top_k=40, top_p=0.9,
                 seed=1, max_tokens=512, lib_path=None, **_unused):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not mmproj_path:
            raise ValueError("mmproj_path is required for MLLM")
        self.max_tokens = int(max_tokens)
        self._lib_path = _find_lib(lib_path, env_var="FLASHRT_MLLM_LIB")
        self._lib = ctypes.CDLL(self._lib_path)

        self._lib.frt_llama_cpp_default_engine_factory.argtypes = []
        self._lib.frt_llama_cpp_default_engine_factory.restype = ctypes.c_void_p
        self._lib.frt_llama_cpp_mllm_runtime_open_with_engine_factory.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.frt_llama_cpp_mllm_runtime_open_with_engine_factory.restype = (
            ctypes.c_int)

        config = {
            "model_family": "mllm",
            "model_path": str(checkpoint),
            "mmproj_path": str(mmproj_path),
            "backend": backend,
            "n_ctx": int(n_ctx),
            "n_threads": int(n_threads),
            "temp": float(temp),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "seed": int(seed),
            "max_tokens": int(max_tokens),
        }
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
        rc = self._lib.frt_llama_cpp_mllm_runtime_open_with_engine_factory(
            config_json, factory_ptr, ctypes.byref(model_ptr))
        if rc != 0 or not model_ptr.value:
            err = factory.last_error(factory.self) or b""
            raise RuntimeError(
                f"frt_llama_cpp_mllm_runtime_open_with_engine_factory failed "
                f"(rc={rc}): {(err.decode(errors='replace') if err else 'no error')}")

        self._model = FrtModelRuntimeV2.from_address(model_ptr.value)
        if self._model.abi_version != _FRT_MODEL_RUNTIME_ABI_VERSION_V2:
            raise RuntimeError(
                f"frt_model_runtime_v2 abi_version={self._model.abi_version}, "
                f"expected {_FRT_MODEL_RUNTIME_ABI_VERSION_V2}.")
        if self._model.struct_size < ctypes.sizeof(FrtModelRuntimeV2):
            raise RuntimeError(
                f"frt_model_runtime_v2 struct_size={self._model.struct_size} "
                f"< ctypes sizeof {ctypes.sizeof(FrtModelRuntimeV2)}.")
        self._model_ptr = model_ptr

    def _make_image_views(self, images):
        """Convert a list of numpy RGB uint8 arrays to frt_image_view[]."""
        n = len(images)
        Views = FrtImageView * n
        views = Views()
        self._rgb_bufs = []
        for i, img in enumerate(images):
            arr = np.ascontiguousarray(img, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(
                    f"Image {i} must be HWC RGB uint8, got shape {arr.shape}")
            self._rgb_bufs.append(arr)
            views[i].struct_size = ctypes.sizeof(FrtImageView)
            views[i].pixel_format = FRT_RT_PIXEL_RGB8
            views[i].data = arr.ctypes.data
            views[i].bytes = arr.nbytes
            views[i].width = arr.shape[1]
            views[i].height = arr.shape[0]
            views[i].stride_bytes = 0
            views[i].reserved = 0
            views[i].timestamp_ns = 0
        return views

    def generate(self, images, prompt):
        """Run one multimodal completion.

        Args:
            images: list of numpy arrays (H,W,3) uint8 RGB. May be empty for
                    text-only (though a pure LLM frontend is more efficient).
            prompt: text prompt (str or bytes). Raw — caller applies chat
                    template. The engine injects media markers for each image.

        Returns:
            Generated text (str).
        """
        if self._model is None:
            raise RuntimeError("MllmJetsonPiFrontend is closed")
        v2 = self._model.verbs_v2
        self_ = self._model.self

        views = self._make_image_views(images)
        rc = v2.set_input(self_, PORT_IMAGES, ctypes.byref(views),
                          ctypes.sizeof(views), -1)
        self._check(rc, "set_input images")

        if isinstance(prompt, str):
            prompt_bytes = prompt.encode("utf-8")
        else:
            prompt_bytes = bytes(prompt)
        rc = v2.set_input(self_, PORT_PROMPT, prompt_bytes,
                          len(prompt_bytes), -1)
        self._check(rc, "set_input prompt")

        rc = v2.run_stage(self_, STAGE_INFER, -1)
        self._check(rc, "run_stage infer")

        cap = self.max_tokens * 8
        out = (ctypes.c_char * cap)()
        written = ctypes.c_uint64(0)
        rc = v2.get_output(self_, PORT_TEXT, out, cap,
                           ctypes.byref(written), -1)
        self._check(rc, "get_output text")
        return out.raw[:written.value].decode("utf-8", errors="replace")

    def infer(self, observation, debug=False):
        """VLAModel-predict-parity entry.

        observation keys: 'images' (list of HWC uint8 arrays), 'prompt' (str).
        Returns: {'text': str}.
        """
        _ = debug
        if not isinstance(observation, dict):
            raise ValueError("observation must be a dict")
        images = observation.get("images", [])
        prompt = observation.get("prompt")
        if prompt is None:
            raise ValueError("observation['prompt'] is required")
        return {"text": self.generate(images, prompt)}

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
