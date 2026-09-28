# SPDX-License-Identifier: Apache-2.0
"""A request that BatchGenerator.insert refuses fails alone.

mlx-lm's insert raises ValueError for a row it cannot accept (max_tokens <= 0,
an empty prompt). It runs after the request's full prefill, inside step(), so
before this fix the exception reached engine_core's loop, whose recovery calls
scheduler.fail_all_requests() and fails every request on the engine. The
scheduler now refuses max_tokens < 1 at admission and turns an insert
ValueError into an error output for that request alone.
"""

from collections import deque
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.engine_core import EngineCore
from omlx.exceptions import InvalidRequestError
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler, _PrefillState

_REFUSAL = "Sequence 0's max_tokens must be > 0."


def _insert_refusing_non_positive(prompts, **kwargs):
    """Stand-in for BatchGenerator.insert with mlx-lm's validate-first check."""
    if kwargs["max_tokens"][0] <= 0:
        raise ValueError(_REFUSAL)
    return [42]


def _make_scheduler(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.side_effect = _insert_refusing_non_positive
    scheduler.batch_generator.next_generated.return_value = []
    scheduler._ensure_batch_generator = MagicMock()
    scheduler._build_sampler_and_processors = MagicMock(return_value=(MagicMock(), []))
    scheduler._build_state_machine = MagicMock(return_value=MagicMock())
    scheduler._preflight_memory_check = MagicMock(return_value=None)
    scheduler._validate_cache = MagicMock(return_value=True)
    return scheduler


def _queue_cache_hit(scheduler, request_id, max_tokens):
    """Queue a request whose prompt is cached up to its last token, so
    _schedule_waiting inserts it without running a prefill."""
    request = Request(
        request_id=request_id,
        prompt=[11, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    request.prompt_token_ids = [11, 12, 13, 14]
    request.num_prompt_tokens = 4
    request.remaining_tokens = [14]
    request.cached_tokens = 3
    request.prompt_cache = [MagicMock()]
    scheduler.waiting.append(request)
    scheduler.requests[request_id] = request
    return request


def _add_running(scheduler, request_id, uid):
    request = Request(
        request_id=request_id,
        prompt=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        status=RequestStatus.RUNNING,
        batch_uid=uid,
    )
    scheduler.running[request_id] = request
    scheduler.requests[request_id] = request
    scheduler.request_id_to_uid[request_id] = uid
    scheduler.uid_to_request_id[uid] = request_id
    return request


def _assert_refused(output, request_id):
    assert output.request_id == request_id
    assert output.finished is True
    assert output.finish_reason == "error"
    assert output.error == _REFUSAL


def test_mlx_lm_insert_refuses_bad_rows_before_touching_batch_state():
    """The isolation relies on insert validating every row before it
    changes batch state; pin that for the installed mlx-lm."""
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.llama import Model, ModelArgs

    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=64,
        )
    )
    generator = BatchGenerator(model, max_tokens=4)
    try:
        for bad in (0, -5):
            with pytest.raises(ValueError, match="max_tokens must be > 0"):
                generator.insert([[1, 2, 3]], max_tokens=[bad])
        with pytest.raises(ValueError, match="empty prompt"):
            generator.insert([[]], max_tokens=[4])
        assert generator._uid_count == 0
        assert not generator._unprocessed_sequences
        assert generator.insert([[1, 2, 3]], max_tokens=[4]) == [0]
    finally:
        generator.close()


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_add_request_refuses_non_positive_max_tokens(
    max_tokens, mock_model, mock_tokenizer
):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    request = Request(
        request_id="bad",
        prompt=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )

    with pytest.raises(InvalidRequestError, match="max_tokens must be at least 1"):
        scheduler.add_request(request)

    assert scheduler.requests == {}
    assert list(scheduler.waiting) == []


def test_add_request_accepts_one_max_token(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    request = Request(
        request_id="one",
        prompt=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
    )

    scheduler.add_request(request)

    assert list(scheduler.waiting) == [request]


def test_refused_insert_fails_only_that_request(mock_model, mock_tokenizer):
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    live = _add_running(scheduler, "live", uid=7)
    _queue_cache_hit(scheduler, "bad", max_tokens=0)
    good = _queue_cache_hit(scheduler, "good", max_tokens=4)
    # Admit both in one pass; decode fairness would otherwise defer "good"
    # to the next step behind the running "live" decode.
    scheduler._prefill_gate_open = MagicMock(return_value=True)

    scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == [good]
    assert len(rejected) == 1
    _assert_refused(rejected[0], "bad")
    assert "bad" not in scheduler.requests
    assert "bad" not in scheduler.running
    assert "bad" not in scheduler.request_id_to_uid
    assert scheduler.running == {"live": live, "good": good}
    assert scheduler.request_id_to_uid == {"live": 7, "good": 42}


def test_refused_chunked_insert_fails_only_that_request(mock_model, mock_tokenizer):
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    live = _add_running(scheduler, "live", uid=7)
    states = {}
    for rid, max_tokens in (("bad", 0), ("good", 4)):
        request = Request(
            request_id=rid,
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=max_tokens),
        )
        request.prompt_token_ids = [11, 12, 13, 14]
        request.num_prompt_tokens = 4
        states[rid] = _PrefillState(
            request=request,
            cache=[MagicMock()],
            tokens_remaining=mx.array([[]]),
            last_token=[14],
            tokens_processed=3,
            base_size=0,
            emitted_boundaries={},
            boundary_enabled=False,
            block_size=0,
            total_length=4,
            sampler=MagicMock(),
            sm=MagicMock(),
            per_row_lps=[],
        )
        scheduler.requests[rid] = request
        scheduler.prefilling.append(request)
        scheduler._prefill_states[rid] = states[rid]
    scheduler._prefill_gate_open = MagicMock(return_value=True)
    scheduler._step_prefill_chunk = MagicMock(return_value=True)
    scheduler._emit_final_boundary_if_needed = MagicMock()

    scheduled, rejected = [], []
    scheduler._advance_chunked_prefills(scheduled, rejected)

    good = states["good"].request
    assert scheduled == [good]
    assert len(rejected) == 1
    _assert_refused(rejected[0], "bad")
    assert "bad" not in scheduler.requests
    assert "bad" not in scheduler._prefill_states
    assert list(scheduler.prefilling) == []
    assert scheduler.running == {"live": live, "good": good}


def test_step_reports_refused_insert_without_raising(mock_model, mock_tokenizer):
    """step() returns the refusal as that request's output; nothing reaches
    the engine loop's fail-all recovery."""
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    live = _add_running(scheduler, "live", uid=7)
    _queue_cache_hit(scheduler, "bad", max_tokens=-5)

    with patch("omlx.scheduler._sync_and_clear_cache"):
        output = scheduler.step()

    refusals = [o for o in output.outputs if o.request_id == "bad"]
    assert len(refusals) == 1
    _assert_refused(refusals[0], "bad")
    assert all(
        o.finish_reason != "error" for o in output.outputs if o is not refusals[0]
    )
    assert scheduler.running == {"live": live}
    assert scheduler.waiting == deque()
    assert "bad" not in scheduler.requests
    assert scheduler.has_requests() is True


@pytest.mark.asyncio
async def test_engine_add_request_refuses_non_positive_max_tokens_alone(
    mock_model, mock_tokenizer
):
    """The refusal reaches only its caller, as the typed 400 error, and leaves
    no tracking behind; a request already admitted is untouched."""
    with patch("omlx.engine_core.get_registry") as mock_registry:
        mock_registry.return_value.acquire.return_value = True
        engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
        try:
            engine.scheduler.fail_all_requests = MagicMock(return_value=[])
            kept = await engine.add_request(
                prompt="Hello", sampling_params=SamplingParams(max_tokens=8)
            )

            with pytest.raises(InvalidRequestError) as refused:
                await engine.add_request(
                    prompt="Hello", sampling_params=SamplingParams(max_tokens=0)
                )

            assert refused.value.field == "max_tokens"
            assert set(engine._output_collectors) == {kept}
            assert set(engine._stream_states) == {kept}
            assert set(engine.scheduler.requests) == {kept}
            engine.scheduler.fail_all_requests.assert_not_called()
        finally:
            engine.close()
