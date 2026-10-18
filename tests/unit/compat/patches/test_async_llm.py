# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import asyncio
import logging
import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.compat.patches import async_llm as patch


@pytest.fixture
def runtime(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.26.0"
    vllm.envs = SimpleNamespace(VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=128)
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(is_cuda=lambda: True)
    module = types.ModuleType("vllm.v1.engine.async_llm")
    module.IterationStats = SimpleNamespace
    module.logger = logging.getLogger(__name__)
    module.AsyncLLM = type("AsyncLLM", (), {"_run_output_handler": lambda self: None})
    for name, value in [
        ("vllm", vllm),
        ("vllm.platforms", platforms),
        ("vllm.v1.engine.async_llm", module),
    ]:
        monkeypatch.setitem(sys.modules, name, value)
    return vllm, platforms, module


@pytest.mark.parametrize(
    "role,cuda",
    [("attention", True), ("ffn", True), (None, True), ("attention", False)],
)
@pytest.mark.parametrize("batch", [0, 1, 128, 129, 256])
def test_output_order_fairness_stats_and_abort(runtime, role, cuda, batch):
    _, platforms, _ = runtime
    platforms.current_platform.is_cuda = lambda: cuda

    async def exercise():
        queue = asyncio.Queue()
        seen, snapshots, stats, records, aborts = [], [], [], [], []
        for sequence in range(4):
            queue.put_nowait(
                SimpleNamespace(
                    outputs=list(range(sequence * batch, (sequence + 1) * batch)),
                    timestamp=sequence,
                    scheduler_stats=sequence,
                    engine_index=0,
                )
            )

        async def abort(ids):
            aborts.extend(ids)

        def process(outputs, timestamp, iteration_stats):
            seen.extend(outputs)
            assert iteration_stats is not None
            return SimpleNamespace(request_outputs=[], reqs_to_abort=[timestamp])

        instance = SimpleNamespace(
            vllm_config=SimpleNamespace(
                additional_config={
                    "afd": {"role": role, "connector": "P2pNcclAFDConnector"}
                }
                if role
                else {}
            ),
            output_handler=None,
            engine_core=SimpleNamespace(
                get_output_async=queue.get, abort_requests_async=abort
            ),
            output_processor=SimpleNamespace(
                process_outputs=process,
                update_scheduler_stats=stats.append,
                propagate_error=lambda error: pytest.fail(str(error)),
            ),
            log_stats=True,
            logger_manager=SimpleNamespace(record=lambda **data: records.append(data)),
            renderer=SimpleNamespace(stat_mm_cache=lambda: None),
        )
        patch._run_output_handler(instance)
        first_task = instance.output_handler
        patch._run_output_handler(instance)
        assert instance.output_handler is first_task

        async def observer():
            snapshots.append(len(stats))

        await asyncio.create_task(observer())
        while len(stats) < 4:
            await asyncio.sleep(0)
        instance.output_handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await instance.output_handler
        assert seen == list(range(4 * batch))
        assert stats == list(range(4))
        assert [r["scheduler_stats"] for r in records] == stats
        assert aborts == [n for n in range(4) for _ in range((batch + 127) // 128)]
        if batch <= 128:
            assert (
                snapshots[0] < 4 if role == "attention" and cuda else snapshots[0] == 4
            )

    asyncio.run(exercise())


def test_exception_propagation(runtime):
    async def exercise():
        error = ValueError("engine failed")
        failures = []

        async def get_output():
            raise error

        instance = SimpleNamespace(
            vllm_config=SimpleNamespace(additional_config={}),
            output_handler=None,
            engine_core=SimpleNamespace(get_output_async=get_output),
            output_processor=SimpleNamespace(propagate_error=failures.append),
            log_stats=False,
            logger_manager=None,
            renderer=None,
        )
        patch._run_output_handler(instance)
        await instance.output_handler
        assert failures == [error]

    asyncio.run(exercise())


def test_exact_version_and_idempotence(runtime):
    vllm, _, module = runtime
    original = module.AsyncLLM._run_output_handler
    vllm.__version__ = "0.30.0"
    assert not patch.apply_async_llm_patch()
    assert module.AsyncLLM._run_output_handler is original
    vllm.__version__ = "0.26.0"
    assert patch.apply_async_llm_patch()
    assert patch.apply_async_llm_patch()
    assert module.AsyncLLM._run_output_handler is patch._run_output_handler
