import asyncio
import base64
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from miles.rollout.verifiers_v1_rollout import (
    MilesSGLangRendererV1Client,
    VerifiersV1RolloutFn,
    _base_sampling_params,
    _build_sglang_generate_payload,
    _decode_replay_array,
    _finish_reason,
    _GenerationCapture,
    _sampling_params,
    trace_to_sample,
    trace_to_samples,
)
from miles.utils.types import Sample


def _args(reward_key: str | None = None, **overrides) -> Namespace:
    values = dict(
        reward_key=reward_key,
        rollout_temperature=1.0,
        rollout_top_p=0.9,
        rollout_top_k=50,
        rollout_max_response_len=128,
        rollout_stop=["</answer>"],
        rollout_stop_token_ids=[2],
        rollout_skip_special_tokens=True,
        lora_rank=0,
        lora_adapter_path=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_opd=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
        sglang_speculative_algorithm=None,
        num_layers=2,
        moe_router_topk=2,
    )
    values.update(overrides)
    return Namespace(**values)


def _node(*, sampled: bool, content: str):
    return SimpleNamespace(sampled=sampled, message=SimpleNamespace(content=content, reasoning_content=None))


def _branch(
    *,
    index: int = 0,
    token_ids: list[int] | None = None,
    sampled_mask: list[bool] | None = None,
    logprobs: list[float] | None = None,
):
    return SimpleNamespace(
        index=index,
        token_ids=[10, 11, 20, 21, 22] if token_ids is None else token_ids,
        sampled_mask=[False, False, True, False, True] if sampled_mask is None else sampled_mask,
        logprobs=[0.0, 0.0, -0.1, 0.0, -0.2] if logprobs is None else logprobs,
        nodes=[
            _node(sampled=False, content="solve this"),
            _node(sampled=True, content="first"),
            _node(sampled=False, content="tool"),
            _node(sampled=True, content="answer"),
        ],
        multi_modal_data=None,
    )


def _trace(**overrides):
    values = dict(
        id="trace-1",
        branches=[_branch()],
        task=SimpleNamespace(data=SimpleNamespace(prompt="solve this", idx="task-1")),
        rewards={"score": 1.25, "bonus": 0.75},
        metrics={"turns": 2.0},
        stop_condition="done",
        error=None,
        has_error=False,
        is_truncated=False,
        reward=2.0,
        last_reply="answer",
        timing=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _capture() -> _GenerationCapture:
    return _GenerationCapture(
        prompt_ids=[10, 11, 20, 21],
        completion_ids=[22],
        completion_logprobs=[-0.2],
        output_text="raw answer",
        sampled_mask=[False, False, True, False, True],
        logprobs=[0.0, 0.0, -0.1, 0.0, -0.2],
        top_logprobs=[[], [], [[-0.1, 20]], [], [[-0.2, 22]]],
        meta_info={"weight_version": "7", "cached_tokens": 3, "prompt_tokens": 4},
        routed_experts=np.arange(16, dtype=np.int32).reshape(4, 2, 2),
        indexer_topk=np.arange(12, dtype=np.int32).reshape(4, 1, 3),
        multimodal_inputs={"images": ["https://example.test/image.png"]},
        multi_modal_data=None,
    )


def test_trace_to_sample_uses_verifiers_branch_suffix_for_training_fields():
    sample = trace_to_sample(_args(), _trace(), group_index=3, index=9)

    assert sample.group_index == 3
    assert sample.index == 9
    assert sample.prompt == "solve this"
    assert sample.tokens == [10, 11, 20, 21, 22]
    assert sample.response == "firsttoolanswer"
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, 0.0, -0.2]
    assert sample.reward == 2.0
    assert sample.status == Sample.Status.COMPLETED
    assert sample.metadata["verifiers_v1_trace_id"] == "trace-1"
    sample.validate()


def test_trace_to_sample_preserves_reward_dict_when_reward_key_is_enabled():
    args = _args(reward_key="score")
    sample = trace_to_sample(args, _trace(), group_index=0, index=0)

    assert sample.reward == {"score": 1.25, "bonus": 0.75, "reward": 2.0}
    assert sample.get_reward_value(args) == 1.25


def test_trace_to_sample_preserves_falsy_labels():
    trace = _trace(task=SimpleNamespace(data=SimpleNamespace(prompt="solve this", idx="task-1", label=0)))

    sample = trace_to_sample(_args(), trace, group_index=0, index=0)

    assert sample.label == 0


def test_trace_to_sample_marks_error_before_truncation():
    error = SimpleNamespace(model_dump=lambda **_kwargs: {"type": "ProviderError"})
    trace = _trace(error=error, has_error=True, is_truncated=True)

    sample = trace_to_sample(_args(), trace, group_index=0, index=0)

    assert sample.status == Sample.Status.FAILED
    assert sample.metadata["verifiers_v1_error"] == {"type": "ProviderError"}


def test_trace_to_samples_preserves_every_graph_branch_with_unique_indices():
    trace = _trace(branches=[_branch(index=0), _branch(index=1)])

    samples = trace_to_samples(_args(), trace, group_index=4, index_start=10)

    assert [sample.index for sample in samples] == [10, 11]
    assert [sample.group_index for sample in samples] == [4, 4]
    assert [sample.metadata["verifiers_v1_branch_index"] for sample in samples] == [0, 1]


def test_convert_group_can_preserve_empty_trace_positions_for_eval_accounting():
    adapter = object.__new__(VerifiersV1RolloutFn)
    adapter.args = _args()
    adapter.client = SimpleNamespace(pop_captures=lambda _trace_id: [])
    adapter._next_sample_index = 0

    group = adapter._convert_group(
        [_trace(id="trace-ok"), _trace(id="trace-empty", branches=[])],
        group_index=2,
        preserve_empty=True,
    )

    assert isinstance(group[0], Sample)
    assert group[0].group_index == 2
    assert group[1] is None


def test_rejected_group_still_drains_captures_for_every_trace():
    drained = []
    adapter = object.__new__(VerifiersV1RolloutFn)
    adapter.args = _args()
    adapter.client = SimpleNamespace(pop_captures=lambda trace_id: drained.append(trace_id) or [])
    adapter._next_sample_index = 0

    group = adapter._convert_group(
        [_trace(id="trace-empty", branches=[]), _trace(id="trace-ok")],
        group_index=2,
    )

    assert group == []
    assert drained == ["trace-empty", "trace-ok"]


@pytest.mark.asyncio
async def test_cancel_pending_aborts_sglang_workers(monkeypatch):
    aborted = []

    async def get_worker_urls(_args, _model):
        return ["http://worker-0", "http://worker-1"]

    async def post(url, payload):
        aborted.append((url, payload))

    monkeypatch.setattr(
        "miles.rollout.verifiers_v1_rollout._sglang_worker_urls",
        get_worker_urls,
    )
    monkeypatch.setattr("miles.utils.http_utils.post", post)
    adapter = object.__new__(VerifiersV1RolloutFn)
    adapter.args = _args(sglang_model_routers=None)
    adapter.model = "test/model"
    future = asyncio.create_task(asyncio.sleep(60))

    await adapter._cancel_pending([future])

    assert future.cancelled()
    assert aborted == [
        ("http://worker-0/abort_request", {"abort_all": True}),
        ("http://worker-1/abort_request", {"abort_all": True}),
    ]


@pytest.mark.asyncio
async def test_train_uses_over_sampling_batch_size_as_refill_granularity(monkeypatch):
    submitted = []

    async def configure_sglang(_args):
        return None

    import miles.utils as miles_utils

    dumper_utils = SimpleNamespace(configure_sglang=configure_sglang)
    monkeypatch.setitem(sys.modules, "miles.utils.dumper_utils", dumper_utils)
    monkeypatch.setattr(miles_utils, "dumper_utils", dumper_utils, raising=False)

    class Environment:
        @asynccontextmanager
        async def serving(self):
            yield

    adapter = object.__new__(VerifiersV1RolloutFn)
    adapter.args = _args(
        rollout_batch_size=1,
        over_sampling_batch_size=3,
        rollout_seed=11,
        n_samples_per_prompt=1,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )
    adapter.env = Environment()
    adapter.max_concurrent = 3
    adapter.dynamic_filter = None
    adapter.data_source = None
    adapter._next_train_task_idx = 0
    adapter._next_group_index = 0
    adapter._next_sample_index = 0
    adapter._task = lambda idx: idx

    async def run_task_group(task, _n, _semaphore, _seed_base):
        submitted.append(task)
        await asyncio.sleep(0)
        return [
            SimpleNamespace(
                id=f"trace-{task}",
                reward=1.0,
                has_error=False,
                is_truncated=False,
                num_turns=1,
                branches=[object()],
            )
        ]

    def convert_group(_traces, *, group_index):
        index = adapter._next_sample_index
        adapter._next_sample_index += 1
        return [
            Sample(
                group_index=group_index,
                index=index,
                tokens=[1],
                response="x",
                response_length=1,
                reward=1.0,
                loss_mask=[1],
                rollout_log_probs=[-0.1],
                status=Sample.Status.COMPLETED,
            )
        ]

    async def no_op(*_args, **_kwargs):
        return None

    async def cancel_pending(futures):
        for future in futures:
            future.cancel()
        await asyncio.gather(*futures, return_exceptions=True)

    adapter._run_task_group = run_task_group
    adapter._convert_group = convert_group
    adapter._apply_miles_rewards = no_op
    adapter._postprocess_train_samples = no_op
    adapter._cancel_pending = cancel_pending

    output = await adapter._call_train(SimpleNamespace())

    assert submitted == [0, 1, 2]
    assert len(output.samples) == 1


def test_exact_capture_preserves_replay_opd_multimodal_and_weight_metadata():
    args = _args(
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=True,
        use_opd=True,
        opd_log_prob_top_k=1,
    )

    sample = trace_to_samples(
        args,
        _trace(),
        captures=[_capture()],
        group_index=0,
        index_start=0,
    )[0]

    np.testing.assert_array_equal(sample.rollout_routed_experts, _capture().routed_experts)
    np.testing.assert_array_equal(sample.rollout_indexer_topk, _capture().indexer_topk)
    assert sample.metadata["opd_student_top_logprobs"] == [[[-0.1, 20]], [], [[-0.2, 22]]]
    assert sample.multimodal_inputs == {"images": ["https://example.test/image.png"]}
    assert sample.weight_versions == ["7"]
    assert sample.response == "firsttoolraw answer"
    assert sample.prefix_cache_info.cached_tokens == 3
    sample.validate()


def test_single_branch_streamed_trace_uses_miles_token_capture():
    trace = _trace(branches=[_branch(token_ids=[], sampled_mask=[], logprobs=[])])

    sample = trace_to_samples(
        _args(),
        trace,
        captures=[_capture()],
        group_index=0,
        index_start=0,
    )[0]

    assert sample.tokens == _capture().sequence_ids
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, 0.0, -0.2]


def test_capture_history_preserves_prior_sampled_token_logprobs():
    client = object.__new__(MilesSGLangRendererV1Client)
    client._captures = {}
    client._record_capture(
        "trace-1",
        prompt_ids=[10],
        completion_ids=[20],
        completion_logprobs=[-0.1],
        output_text="first raw",
        output_top_logprobs=[[[-0.1, 20]]],
        meta_info={},
        routed_experts=None,
        indexer_topk=None,
        multimodal_inputs={},
        multi_modal_data=None,
    )
    client._record_capture(
        "trace-1",
        prompt_ids=[10, 20, 30],
        completion_ids=[40],
        completion_logprobs=[-0.2],
        output_text="second raw",
        output_top_logprobs=[[[-0.2, 40]]],
        meta_info={},
        routed_experts=None,
        indexer_topk=None,
        multimodal_inputs={},
        multi_modal_data=None,
    )

    capture = client.pop_captures("trace-1")[-1]
    assert capture.sampled_mask == [False, True, False, True]
    assert capture.logprobs == [0.0, -0.1, 0.0, -0.2]
    assert capture.top_logprobs == [[], [[-0.1, 20]], [], [[-0.2, 40]]]


def test_replay_requires_an_exact_capture_instead_of_silently_dropping_data():
    with pytest.raises(ValueError, match="exact Miles capture for routing replay"):
        trace_to_sample(
            _args(use_rollout_routing_replay=True),
            _trace(),
            group_index=0,
            index=0,
        )


class _Sampling:
    def __init__(self, **values):
        self.values = values

    def model_dump(self, exclude_none=True):
        if not exclude_none:
            return dict(self.values)
        return {key: value for key, value in self.values.items() if value is not None}


class _SamplingModel:
    @classmethod
    def model_validate(cls, values):
        return values


def test_eval_sampling_uses_miles_eval_values_only_as_config_defaults():
    adapter = object.__new__(VerifiersV1RolloutFn)
    adapter.config = SimpleNamespace(
        sampling=_Sampling(
            temperature=0.3,
            extra_body={"min_new_tokens": 4},
        )
    )
    eval_args = _args(
        rollout_temperature=0.1,
        rollout_top_p=0.8,
        rollout_top_k=12,
        rollout_max_response_len=48,
    )

    sampling = adapter._sampling_config(_SamplingModel, eval_args, min_new_tokens=2)

    assert sampling == {
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 12,
        "max_tokens": 48,
        "extra_body": {"min_new_tokens": 4},
    }


def test_sampling_params_translate_verifiers_max_tokens_to_sglang_generate_shape():
    sampling = _sampling_params(_args(), _Sampling(temperature=0.2, max_tokens=32), stop_token_ids=[99])

    assert sampling == {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 32,
        "stop": ["</answer>"],
        "stop_token_ids": [99, 2],
        "skip_special_tokens": True,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


def test_v020_sampling_fallback_honors_request_fields_and_extra_body():
    dialect = SimpleNamespace()
    sampling, options = _base_sampling_params(
        dialect,
        {"temperature": 0.3, "max_completion_tokens": 24, "seed": 17},
        _Sampling(extra_body={"chat_template_kwargs": {"enable_thinking": False}, "min_p": 0.1}),
    )

    assert sampling == {"temperature": 0.3, "max_new_tokens": 24, "sampling_seed": 17, "min_p": 0.1, "n": 1}
    assert options["chat_template_kwargs"] == {"enable_thinking": False}


def test_sampling_honors_native_sglang_fields_and_request_options_from_sdk_body():
    sampling, options = _base_sampling_params(
        SimpleNamespace(),
        {
            "regex": r"[0-9]+",
            "min_new_tokens": 4,
            "priority": -2,
            "cache_salt": "request-salt",
            "chat_template_kwargs": {"enable_thinking": True},
        },
        _Sampling(
            extra_body={
                "priority": 9,
                "cache_salt": "config-salt",
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ),
    )

    assert sampling == {"regex": r"[0-9]+", "min_new_tokens": 4, "n": 1}
    assert options == {
        "priority": -2,
        "cache_salt": "request-salt",
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_build_sglang_generate_payload_requests_all_miles_sidecars():
    payload = _build_sglang_generate_payload(
        _args(
            use_rollout_routing_replay=True,
            use_rollout_indexer_replay=True,
            use_opd=True,
            opd_log_prob_top_k=8,
        ),
        prompt_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 8},
        multimodal_inputs={"images": ["data:image/png;base64,abc"]},
        multi_modal_data=SimpleNamespace(mm_hashes={"image": ["hash"]}),
        request_options={"priority": 2, "cache_salt": "salt"},
    )

    assert payload == {
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 8},
        "return_logprob": True,
        "return_routed_experts": True,
        "routed_experts_start_len": 0,
        "return_indexer_topk": True,
        "top_logprobs_num": 8,
        "image_data": ["data:image/png;base64,abc"],
        "mm_hashes": ["hash"],
        "priority": 2,
        "extra_key": "salt",
    }


def test_finish_reason_promotes_valid_tool_calls():
    status = SimpleNamespace(OK="ok")
    tool_call = SimpleNamespace(status="ok")
    output = {"meta_info": {"finish_reason": {"type": "stop"}}}

    assert _finish_reason(output, [tool_call], status) == "tool_calls"


def test_finish_reason_leaves_malformed_tool_calls_as_stop():
    status = SimpleNamespace(OK="ok")
    tool_call = SimpleNamespace(status="invalid_json")
    output = {"meta_info": {"finish_reason": {"type": "stop"}}}

    assert _finish_reason(output, [tool_call], status) == "stop"


def test_replay_fixture_uses_real_base64_int32_wire_shape():
    capture = _capture()
    encoded = base64.b64encode(capture.routed_experts.tobytes()).decode("ascii")
    assert np.frombuffer(base64.b64decode(encoded), dtype=np.int32).reshape(4, 2, 2).tolist() == (
        capture.routed_experts.tolist()
    )


def test_replay_payload_rejects_values_for_an_empty_token_range():
    encoded = base64.b64encode(np.array([1], dtype=np.int32).tobytes()).decode("ascii")

    with pytest.raises(ValueError, match="empty token range"):
        _decode_replay_array(encoded, rows=0, layers=1, topk=1)
