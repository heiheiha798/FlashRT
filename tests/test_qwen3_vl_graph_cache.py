"""Regression tests for Qwen3-VL CUDA Graph cache management.

These tests avoid checkpoints and real CUDA Graph capture. They patch the small
``torch.cuda`` surface used by ``Qwen3VlTorchFrontendRtx._ensure_decode_graph``
so the test can drive graph-cache bookkeeping in CI.
"""
from __future__ import annotations

import collections
import contextlib


def test_qwen3_vl_decode_graph_cache_is_lru_bounded(monkeypatch):
    """Distinct decode positions must not grow the graph cache forever."""
    import torch

    from flash_rt.frontends.torch.qwen3_vl_rtx import Qwen3VlTorchFrontendRtx

    class _FakeStream:
        cuda_stream = 0

        def wait_stream(self, _other):
            pass

        def synchronize(self):
            pass

    class _FakeGraph:
        def replay(self):
            pass

    class _FakeLLM:
        def __init__(self):
            self._graph_stream = _FakeStream()
            self._static_token_id = object()
            self.calls = []

        def _rope_cos_sin(self, rope_pos):
            return object(), object()

        def forward_own_decode_nvfp4(self, token, cos, sin, cache_pos):
            self.calls.append((token, cos, sin, cache_pos))

    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: _FakeStream(), raising=False)
    monkeypatch.setattr(torch.cuda, "stream",
                        lambda _stream: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "graph",
                        lambda _graph, stream=None: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph",
                        lambda: _FakeGraph(), raising=False)

    fe = Qwen3VlTorchFrontendRtx.__new__(Qwen3VlTorchFrontendRtx)
    fe.llm = _FakeLLM()
    fe.max_decode_graphs = 2
    fe._decode_graphs = collections.OrderedDict()

    for cache_pos in range(5):
        fe._ensure_decode_graph(cache_pos, rope_pos=100 + cache_pos)

    assert len(fe._decode_graphs) == fe.max_decode_graphs
    assert list(fe._decode_graphs) == [3, 4]
