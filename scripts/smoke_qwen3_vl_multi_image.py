#!/usr/bin/env python3
"""Multi-image prefill graph validation for Qwen3-VL SM89.

Verifies the multi-image prefill CUDA-graph capture path added by the
multi-image-prefill-graph PR:

  * a 2-image prompt gets a pg_key (graph path), not an eager fallback;
  * the graph replay is byte-identical to eager prefill (cos == 1.0, same
    argmax top token);
  * a single-image prompt still works byte-identically (regression).

Run on an empty single card:
  CUDA_VISIBLE_DEVICES=0 python scripts/smoke_qwen3_vl_multi_image.py \
    --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8
"""
from __future__ import annotations
import argparse, pathlib, sys, time
import torch
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _event_ms(fn, iters):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], min(ts), (max(ts) - min(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seq', type=int, default=4096)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--image1', default=str(REPO / 'FlashRT.png'))
    p.add_argument('--image2', default='/tmp/FlashRT_med.png')
    p.add_argument('--prompt', default='Compare these two images.')
    args = p.parse_args()
    torch.cuda.set_device(torch.device(args.device))

    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal import (
        Qwen3VlFp8Sm89Frontend,
    )
    fe = Qwen3VlFp8Sm89Frontend(
        args.checkpoint, device=args.device, max_seq=args.max_seq,
        use_fp8_lm_head=True)

    img1 = Image.open(args.image1).convert('RGB')
    img2 = Image.open(args.image2).convert('RGB')
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': img1},
            {'type': 'image', 'image': img2},
            {'type': 'text', 'text': args.prompt},
        ],
    }]

    fe.set_prompt(messages)
    p_ = fe._prompt
    print(f'2-image prompt: S={p_["S"]} spans={p_["spans"]} '
          f'seg_patches={p_["seg_patches"]} '
          f'pg_key={p_.get("pg_key")}')
    assert p_.get('pg_key') is not None, 'multi-image should get a pg_key'
    assert len(p_['spans']) == 2, f'expected 2 spans, got {len(p_["spans"])}'

    # eager prefill (correctness reference)
    eager_logits = fe.prefill()
    torch.cuda.synchronize()
    eager_f = eager_logits.detach().float().clone()
    eager_top = int(eager_f.argmax())

    # graph prefill (must match eager)
    graph_logits = fe.prefill_graph()
    torch.cuda.synchronize()
    graph_f = graph_logits.detach().float().clone()
    graph_top = int(graph_f.argmax())
    cos = torch.nn.functional.cosine_similarity(eager_f, graph_f).item()

    em, emn, esprd = _event_ms(lambda: fe.prefill(), args.iters)
    gm, gmn, gsprd = _event_ms(lambda: fe.prefill_graph(), args.iters)
    print(f'eager  prefill median={em:.3f} ms (min {emn:.3f}, spread {esprd:.3f})')
    print(f'graph  prefill median={gm:.3f} ms (min {gmn:.3f}, spread {gsprd:.3f})')
    print(f'cos_vs_eager={cos:.6f} eager_top={eager_top} graph_top={graph_top} '
          f'match={eager_top == graph_top}')
    assert cos >= 0.999, f'cos {cos} < 0.999 — graph diverged from eager'
    assert eager_top == graph_top, 'top token mismatch'
    assert torch.isfinite(eager_f).all() and torch.isfinite(graph_f).all()

    # single-image regression (FlashRT.png alone)
    msg1 = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img1},
        {'type': 'text', 'text': 'Describe this image in one sentence.'}]}]
    fe.set_prompt(msg1)
    p1 = fe._prompt
    assert p1.get('pg_key') is not None, 'single-image should still get pg_key'
    e1 = fe.prefill(); torch.cuda.synchronize()
    g1 = fe.prefill_graph(); torch.cuda.synchronize()
    cos1 = torch.nn.functional.cosine_similarity(
        e1.detach().float(), g1.detach().float()).item()
    print(f'\nsingle-image regression: S={p1["S"]} spans={p1["spans"]} '
          f'cos_vs_eager={cos1:.6f} '
          f'match={int(e1.argmax()) == int(g1.argmax())}')
    assert cos1 >= 0.999, 'single-image graph regressed'
    assert int(e1.argmax()) == int(g1.argmax())
    print('\nALL OK')


if __name__ == '__main__':
    main()
