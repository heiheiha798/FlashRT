"""End-to-end smoke test for the Jetson-PI multimodal LLM provider through
the Python ``flash_rt.load_model(framework="jetson_pi", config="mllm")`` entry.

Skips (returns early) when FLASHRT_MLLM_MODEL / FLASHRT_MLLM_MMPROJ are unset.

Env:
  FLASHRT_MLLM_MODEL   path to a VLM GGUF (e.g. Qwen2.5-VL-3B-Instruct-q4_0.gguf)
  FLASHRT_MLLM_MMPROJ  path to the VIT mmproj GGUF
  FLASHRT_MLLM_LIB     (optional) path to libflashrt_cpp_llama_cpp_provider_c.so
  FLASHRT_MLLM_BACKEND (optional) backend for the Jetson-PI engine; default
                        "cpu" (byte-identical to the original test). Set to
                        "cuda" to run the real forward pass on the GPU (the
                        engine maps backend=="cuda" to full-layer GPU offload
                        for both the LLM and the VIT/mmproj encoder).
                        CUDA_VISIBLE_DEVICES selects the physical card.
"""

import os
import sys

import numpy as np


def _skip(msg):
    print(f"SKIP - {msg}")
    return 0


def main():
    model_env = os.environ.get("FLASHRT_MLLM_MODEL")
    mmproj_env = os.environ.get("FLASHRT_MLLM_MMPROJ")
    if not model_env or not os.path.exists(model_env):
        return _skip("FLASHRT_MLLM_MODEL not set or missing")
    if not mmproj_env or not os.path.exists(mmproj_env):
        return _skip("FLASHRT_MLLM_MMPROJ not set or missing")

    import flash_rt

    # Default "cpu" keeps the original behavior; set FLASHRT_MLLM_BACKEND=cuda
    # to exercise the real GPU forward pass through the Jetson-PI engine.
    backend = os.environ.get("FLASHRT_MLLM_BACKEND", "cpu") or "cpu"

    fe = flash_rt.load_model(
        model_env,
        framework="jetson_pi",
        config="mllm",
        mmproj_path=mmproj_env,
        backend=backend,
        n_ctx=2048,
        n_threads=0,
        temp=0.0,
        top_k=0,
        top_p=0.0,
        seed=1,
        max_tokens=64,
        lib_path=os.environ.get("FLASHRT_MLLM_LIB"))

    red_image = np.zeros((224, 224, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255

    text = fe.generate([red_image], "Describe this image in one sentence.")

    failed = 0
    def check(cond, msg):
        nonlocal failed
        if cond:
            print(f"ok  : {msg}")
        else:
            print(f"FAIL: {msg}")
            failed = 1

    check(isinstance(text, str) and len(text) > 0, "generated text is non-empty str")
    print(f"    generated: {text!r}")
    if isinstance(text, str) and text:
        printable = any(0x20 <= ord(c) < 0x7f for c in text)
        check(printable, "generated text contains printable chars")

    del fe

    print("\n== JETSON_PI MLLM PYTHON " + ("PASSED" if not failed else "FAILED") + " ==")
    return failed


if __name__ == "__main__":
    sys.exit(main())
