# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fair API output scheduling for the pinned GPU AFD runtime.

The owning layer is AsyncLLM's CPU output consumer, not the GPU scheduler or
connector. Preserve token order, statistics, aborts, and upstream chunking;
only add a cooperative scheduling point. Remove when pinned upstream provides
an unconditional per-message fairness point. Propose this fix upstream after
same-configuration benchmark qualification.
"""

import asyncio

from afd_plugin.compat.vllm import TARGET_VLLM_VERSION


# Upstream source: vLLM v0.26.0, vllm/v1/engine/async_llm.py
# Patch reason: a ready output queue plus sub-chunk batches can starve API tasks.
# Patch functionality: yield after each CUDA AFD Attention output message while
# preserving the full copied upstream path for non-AFD, FFN, and other platforms.
# Signature: matches upstream; no added parameters or return annotation.
def _run_output_handler(self):
    """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

    # ### PATCH START: Lazy runtime bindings and AFD scope
    from vllm import envs
    from vllm.platforms import current_platform
    from vllm.v1.engine.async_llm import IterationStats, logger

    from afd_plugin.config import parse_optional_afd_config

    afd_config = parse_optional_afd_config(self.vllm_config, validate=False)
    fair_output_scheduling = (
        afd_config is not None
        and afd_config.role == "attention"
        and current_platform.is_cuda()
    )
    # ### PATCH END: Lazy runtime bindings and AFD scope

    if self.output_handler is not None:
        return

    # Ensure that the task doesn't have a circular ref back to the AsyncLLM
    # object, or else it won't be garbage collected and cleaned up properly.
    engine_core = self.engine_core
    output_processor = self.output_processor
    log_stats = self.log_stats
    # We use a mutable list for logger_manager so that it can be updated
    # during elastic EP scaling (see scale_elastic_ep) without creating
    # a circular reference via self.
    self._logger_ref = [self.logger_manager]
    logger_ref = self._logger_ref
    renderer = self.renderer
    chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

    async def output_handler():
        try:
            while True:
                # 1) Pull EngineCoreOutputs from the EngineCore.
                outputs = await engine_core.get_output_async()
                num_outputs = len(outputs.outputs)

                iteration_stats = (
                    IterationStats() if (log_stats and num_outputs) else None
                )

                # Split outputs into chunks of at most
                # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                # event loop for too long.
                engine_core_outputs = outputs.outputs
                for start in range(0, num_outputs, chunk_size):
                    end = start + chunk_size
                    outputs_slice = engine_core_outputs[start:end]
                    # 2) Process EngineCoreOutputs.
                    processed_outputs = output_processor.process_outputs(
                        outputs_slice, outputs.timestamp, iteration_stats
                    )
                    # NOTE: RequestOutputs are pushed to their queues.
                    assert not processed_outputs.request_outputs

                    # Allow other asyncio tasks to run between chunks
                    if end < num_outputs:
                        await asyncio.sleep(0)

                    # 3) Abort any reqs that finished due to stop strings.
                    if processed_outputs.reqs_to_abort:
                        await engine_core.abort_requests_async(
                            processed_outputs.reqs_to_abort
                        )

                output_processor.update_scheduler_stats(outputs.scheduler_stats)

                # 4) Logging.
                # TODO(rob): make into a coroutine and launch it in
                # background thread once Prometheus overhead is non-trivial.
                if logger_ref[0]:
                    logger_ref[0].record(
                        engine_idx=outputs.engine_index,
                        scheduler_stats=outputs.scheduler_stats,
                        iteration_stats=iteration_stats,
                        mm_cache_stats=renderer.stat_mm_cache(),
                    )

                # ### PATCH START: Yield after every AFD output message
                # Queue.get() need not suspend when EngineCore output is backlogged.
                # Keep HTTP, metrics, and request admission runnable even when
                # num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE (including zero).
                if fair_output_scheduling:
                    await asyncio.sleep(0)
                # ### PATCH END: Yield after every AFD output message
        except Exception as e:
            logger.exception("AsyncLLM output_handler failed.")
            output_processor.propagate_error(e)

    self.output_handler = asyncio.create_task(output_handler())


def apply_async_llm_patch() -> bool:
    """Install idempotently only on the exact pinned runtime; CPU-safe import."""
    import vllm

    if vllm.__version__ != TARGET_VLLM_VERSION:
        return False
    from vllm.v1.engine.async_llm import AsyncLLM

    AsyncLLM._run_output_handler = _run_output_handler
    return True
