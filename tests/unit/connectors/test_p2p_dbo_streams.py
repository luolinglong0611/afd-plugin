# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.gpu.p2p import P2pNcclAFDConnector  # noqa: E402


def test_dbo_directional_dependencies(monkeypatch):
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            calls.append((self.name, "wait_stream", other.name))

        def wait_event(self, event):
            calls.append((self.name, "wait_event", event))

    class Event:
        def record(self):
            calls.append(("send_done", "record"))

    @contextmanager
    def stream_context(stream):
        calls.append(("enter", stream.name))
        yield
        calls.append(("exit", stream.name))

    compute = Stream("compute")
    connector = object.__new__(P2pNcclAFDConnector)
    connector._overlap_enabled = True
    connector._send_stream = Stream("send")
    connector._receive_stream = Stream("receive")
    event = Event()
    connector._attention_send_done = (Event(), event)
    connector._send_attn_output_current_stream = lambda *args, **kwargs: calls.append(
        ("send",)
    )
    output = SimpleNamespace(device="cuda")

    def receive(*args, **kwargs):
        calls.append(("receive",))
        return output

    connector._recv_ffn_output_current_stream = receive
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: compute)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    tensor = SimpleNamespace(device="cuda")
    context = SimpleNamespace(metadata=SimpleNamespace(stage_idx=1))
    connector.send_attn_output(tensor, context)
    assert connector.recv_ffn_output(tensor, 1) is output
    assert calls == [
        ("send", "wait_stream", "compute"),
        ("enter", "send"),
        ("send",),
        ("send_done", "record"),
        ("exit", "send"),
        ("enter", "receive"),
        ("receive", "wait_event", event),
        ("receive",),
        ("exit", "receive"),
        ("compute", "wait_stream", "receive"),
    ]


def test_no_dbo_keeps_original_path_without_cuda_initialization(monkeypatch):
    connector = object.__new__(P2pNcclAFDConnector)
    connector._overlap_enabled = False
    calls = []
    connector._send_attn_output_current_stream = lambda *args, **kwargs: calls.append(
        "send"
    )
    connector._recv_ffn_output_current_stream = lambda *args, **kwargs: "output"

    def forbidden(*args, **kwargs):
        pytest.fail("non-DBO path accessed a CUDA stream")

    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    connector.send_attn_output("input", "context")
    assert connector.recv_ffn_output("input") == "output"
    assert calls == ["send"]
