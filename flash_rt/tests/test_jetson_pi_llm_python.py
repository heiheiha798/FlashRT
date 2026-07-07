"""End-to-end smoke test for the Jetson-PI generic GGUF LLM provider through
the Python ``flash_rt.load_model(framework="jetson_pi", config="llm")`` entry.

Skips (returns early) when FLASHRT_LLM_MODEL is unset.

Env:
  FLASHRT_LLM_MODEL    path to a GGUF LLM (e.g. qwen3-0.6b-q4_k_m.gguf)
  FLASHRT_LLM_LIB      (optional) path to libflashrt_cpp_llama_cpp_provider_c.so
  FLASHRT_LLM_BACKEND  (optional) backend for the Jetson-PI engine; default
                        "cpu" (byte-identical to the original test). Set to
                        "cuda" to run the real forward pass on the GPU (the
                        engine maps backend=="cuda" to full-layer GPU offload).
                        CUDA_VISIBLE_DEVICES selects the physical card.
"""

import os
import sys


def _skip(msg):
    print(f"SKIP - {msg}")
    return 0


def main():
    model_env = os.environ.get("FLASHRT_LLM_MODEL")
    if not model_env or not os.path.exists(model_env):
        return _skip("FLASHRT_LLM_MODEL not set or missing")

    import flash_rt

    # Default "cpu" keeps the original behavior; set FLASHRT_LLM_BACKEND=cuda
    # to exercise the real GPU forward pass through the Jetson-PI engine.
    backend = os.environ.get("FLASHRT_LLM_BACKEND", "cpu") or "cpu"

    fe = flash_rt.load_model(
        model_env,
        framework="jetson_pi",
        config="llm",
        backend=backend,
        n_ctx=2048,
        n_threads=0,
        temp=0.0,        # greedy for deterministic test output
        top_k=0,
        top_p=0.0,
        seed=1,
        max_tokens=64,
        lib_path=os.environ.get("FLASHRT_LLM_LIB"))

    text = fe.generate("What is 2 plus 2? The answer is")

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

    print("\n== JETSON_PI LLM PYTHON " + ("PASSED" if not failed else "FAILED") + " ==")
    return failed


if __name__ == "__main__":
    sys.exit(main())
