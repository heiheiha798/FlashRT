"""End-to-end smoke test for the Jetson-PI Pi0 provider through the Python
``flash_rt.load_model(framework="jetson_pi", ...)`` entry.

Skips (returns early) when the weights / fixture env vars are unset, so CI
without weights still passes.

Env:
  FLASHRT_PI0_MODEL        path to Pi0 policy GGUF
  FLASHRT_PI0_MMPROJ       path to VIT mmproj GGUF
  FLASHRT_PI0_FIXTURE_DIR  dir with image.png, wrist_image.png, state.bin, prompt.txt
  FLASHRT_PI0_LIB          (optional) path to libflashrt_cpp_llama_cpp_provider_c.so
  FLASHRT_PI0_ACTION_STEPS (optional) override; default 10 (pi0_base).
  FLASHRT_PI0_ACTION_DIM   (optional) override; default 32.

Run from the repo root:
    FLASHRT_PI0_MODEL=... FLASHRT_PI0_MMPROJ=... FLASHRT_PI0_FIXTURE_DIR=/tmp/pi0_fixture \
    LD_LIBRARY_PATH=.../miniconda3/lib:FlashRT/cpp/build-jetson-pi \
    python flash_rt/tests/test_jetson_pi_pi0_python.py
"""

import os
import sys


def _skip(msg):
    print(f"SKIP - {msg}")
    return 0


def main():
    model_env = os.environ.get("FLASHRT_PI0_MODEL")
    mmproj_env = os.environ.get("FLASHRT_PI0_MMPROJ")
    fixture_env = os.environ.get("FLASHRT_PI0_FIXTURE_DIR")
    if not model_env or not os.path.exists(model_env):
        return _skip("FLASHRT_PI0_MODEL not set or missing")
    if not mmproj_env or not os.path.exists(mmproj_env):
        return _skip("FLASHRT_PI0_MMPROJ not set or missing")
    if not fixture_env or not os.path.isdir(fixture_env):
        return _skip("FLASHRT_PI0_FIXTURE_DIR not set or missing")
    for name in ("image.png", "wrist_image.png", "state.bin", "prompt.txt"):
        if not os.path.exists(os.path.join(fixture_env, name)):
            return _skip(f"fixture {name} missing in {fixture_env}")

    import numpy as np
    from PIL import Image

    action_steps = int(os.environ.get("FLASHRT_PI0_ACTION_STEPS", "10"))
    action_dim = int(os.environ.get("FLASHRT_PI0_ACTION_DIM", "32"))

    import flash_rt
    model = flash_rt.load_model(
        model_env,
        framework="jetson_pi",
        mmproj_path=mmproj_env,
        backend="cpu",
        num_views=2,
        action_steps=action_steps,
        action_dim=action_dim,
        lib_path=os.environ.get("FLASHRT_PI0_LIB"))

    image = np.asarray(Image.open(os.path.join(fixture_env, "image.png")).convert("RGB"), dtype=np.uint8)
    wrist = np.asarray(Image.open(os.path.join(fixture_env, "wrist_image.png")).convert("RGB"), dtype=np.uint8)
    with open(os.path.join(fixture_env, "state.bin"), "rb") as f:
        state = np.frombuffer(f.read(), dtype=np.float32)
    if state.size != action_dim:
        print(f"FAIL: state.bin has {state.size} floats, expected {action_dim}")
        return 1
    with open(os.path.join(fixture_env, "prompt.txt")) as f:
        prompt = f.read().rstrip("\n")

    actions = model.predict([image, wrist], prompt=prompt, state=state)

    failed = 0
    def check(cond, msg):
        nonlocal failed
        if cond:
            print(f"ok  : {msg}")
        else:
            print(f"FAIL: {msg}")
            failed = 1

    check(actions.shape == (action_steps, action_dim),
          f"actions shape == ({action_steps},{action_dim}), got {actions.shape}")
    if not np.any(np.isnan(actions)) and not np.any(np.isinf(actions)):
        check(True, "actions contain no NaN/Inf")
    else:
        check(False, "actions contain NaN/Inf")
    check(bool(np.any(actions != 0)), "actions are not all zero")

    del model

    print("\n== JETSON_PI PYTHON " + ("PASSED" if not failed else "FAILED") + " ==")
    return failed


if __name__ == "__main__":
    sys.exit(main())
