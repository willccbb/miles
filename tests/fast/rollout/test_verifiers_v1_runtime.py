import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

if sys.version_info < (3, 11):
    pytest.skip("Verifiers V1 requires Python 3.11+", allow_module_level=True)

pytest.importorskip("verifiers", minversion="0.2.0")
pytest.importorskip("renderers", minversion="0.1.8")

from anthropic.types import RawMessageStreamEvent
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import ResponseCompletedEvent, ResponseIncompleteEvent
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
from verifiers.v1.errors import OverlongPromptError
from verifiers.v1.types import AssistantMessage, Response, SamplingConfig, ToolCall, Usage

from miles.rollout.verifiers_v1_rollout import (
    MilesSGLangRendererV1Client,
    _base_sampling_params,
    _serialize_completion,
    _stream_chunks,
)


def _args(**overrides):
    values = {
        "hf_checkpoint": "test/model",
        "lora_adapter_path": None,
        "lora_rank": 0,
        "moe_router_topk": 1,
        "num_layers": 1,
        "opd_log_prob_top_k": 0,
        "opd_top_k_strategy": "only-student",
        "rollout_max_context_len": 64,
        "rollout_max_response_len": 8,
        "rollout_skip_special_tokens": True,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "rollout_temperature": 0.7,
        "rollout_top_k": None,
        "rollout_top_p": 0.9,
        "sglang_model_routers": None,
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_policy": "round_robin",
        "sglang_router_port": 30000,
        "use_opd": False,
        "use_rollout_indexer_replay": False,
        "use_rollout_routing_replay": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("dialect", [ChatDialect(), ResponsesDialect(), AnthropicDialect()])
def test_completion_and_stream_round_trip_through_published_v1_dialects(dialect):
    response = Response(
        id="response-id",
        created=1,
        model="test/model",
        message=AssistantMessage(
            content="answer",
            reasoning_content="reasoning",
            tool_calls=[ToolCall(id="call-1", name="lookup", arguments='{"query":"x"}')],
        ),
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=3, completion_tokens=2),
    )

    raw = _serialize_completion(response, dialect, "test/model")
    dialect.validate_response(raw)
    parser = dialect.stream_parser()
    for chunk in _stream_chunks(dialect, raw):
        payload = chunk.removeprefix(b"data: ").strip()
        if payload != b"[DONE]":
            event = json.loads(payload)
            if isinstance(dialect, ChatDialect):
                ChatCompletionChunk.model_validate(event)
            elif isinstance(dialect, ResponsesDialect):
                ResponseCompletedEvent.model_validate(event)
            else:
                TypeAdapter(RawMessageStreamEvent).validate_python(event)
        if chunk == b"data: [DONE]\n\n" and parser.on_done is not None:
            parser.on_done()
        parser.feed(chunk)
    parsed = parser.finish()

    assert parsed.message.content == "answer"
    assert parsed.message.reasoning_content == "reasoning"
    assert parsed.message.tool_calls[0].name == "lookup"
    assert parsed.finish_reason == "tool_calls"
    assert parsed.usage.prompt_tokens == 3
    assert parsed.usage.completion_tokens == 2


def test_responses_length_stream_uses_the_official_incomplete_event_schema():
    dialect = ResponsesDialect()
    response = Response(
        id="response-id",
        created=1,
        model="test/model",
        message=AssistantMessage(content="partial"),
        finish_reason="length",
        usage=Usage(prompt_tokens=3, completion_tokens=2),
    )

    raw = _serialize_completion(response, dialect, "test/model")
    event_payload = _stream_chunks(dialect, raw)[0].removeprefix(b"data: ").strip()

    event = ResponseIncompleteEvent.model_validate_json(event_payload)
    assert event.response.status == "incomplete"


@pytest.mark.parametrize(
    ("dialect", "body"),
    [
        (
            ChatDialect(),
            {
                "messages": [{"role": "user", "content": "question"}],
                "max_completion_tokens": 40,
                "temperature": 0.9,
                "seed": 9,
            },
        ),
        (
            ResponsesDialect(),
            {"input": "question", "max_output_tokens": 40, "temperature": 0.9},
        ),
        (
            AnthropicDialect(),
            {
                "messages": [{"role": "user", "content": "question"}],
                "max_tokens": 40,
                "temperature": 0.9,
            },
        ),
    ],
)
def test_sampling_config_and_deterministic_seed_win_in_every_dialect(dialect, body):
    seeded = SamplingConfig(temperature=0.2, max_tokens=8, top_k=7).model_copy(update={"sampling_seed": 123})
    request_body = dialect.apply_overrides(body, "test/model", seeded)

    sampling, _ = _base_sampling_params(dialect, request_body, seeded)

    assert sampling["sampling_seed"] == 123
    assert sampling["temperature"] == 0.2
    assert sampling["max_new_tokens"] == 8
    assert sampling["top_k"] == 7


class _Rendered:
    token_ids = [10, 11]
    multi_modal_data = None
    is_content = [True, True]

    @staticmethod
    def message_token_spans():
        return [(0, 2)]


class _Renderer:
    def render(self, messages, *, tools, add_generation_prompt):
        assert messages == [{"role": "user", "content": "question"}]
        assert tools is None
        assert add_generation_prompt is True
        return _Rendered()

    @staticmethod
    def get_stop_token_ids():
        return [99]

    @staticmethod
    def parse_response(token_ids, *, tools):
        assert token_ids == [20, 21]
        assert tools is None
        return SimpleNamespace(content="answer", reasoning_content=None, tool_calls=[])


@pytest.mark.parametrize(
    ("dialect", "body"),
    [
        (
            ChatDialect(),
            {
                "model": "ignored-by-miles",
                "messages": [{"role": "user", "content": "question"}],
            },
        ),
        (ResponsesDialect(), {"model": "ignored-by-miles", "input": "question"}),
        (
            AnthropicDialect(),
            {
                "model": "ignored-by-miles",
                "max_tokens": 2,
                "messages": [{"role": "user", "content": "question"}],
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_client_uses_real_v020_request_and_response_contract(monkeypatch, dialect, body):
    requests = []

    async def fake_post(url, payload, headers=None):
        requests.append((url, payload, headers))
        return {
            "request_id": "request-id",
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 20], [-0.2, 21]],
            },
        }

    monkeypatch.setattr("miles.utils.http_utils.post", fake_post)
    client = MilesSGLangRendererV1Client(_args(), config_client=SimpleNamespace(), model="test/model")
    client._create_renderer_pool = lambda *_args, **_kwargs: _Renderer()

    response = await client.get_response(
        dialect,
        body,
        "test/model",
        SamplingConfig(temperature=0.2, max_tokens=2),
        session_id="trace-id",
    )

    assert response.message.content == "answer"
    assert response.tokens.prompt_ids == [10, 11]
    assert response.tokens.completion_ids == [20, 21]
    assert response.tokens.completion_logprobs == [-0.1, -0.2]
    dialect.validate_response(response.raw)
    assert requests == [
        (
            "http://127.0.0.1:30000/generate",
            {
                "input_ids": [10, 11],
                "sampling_params": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_new_tokens": 2,
                    "stop_token_ids": [99],
                    "skip_special_tokens": True,
                    "no_stop_trim": True,
                    "spaces_between_special_tokens": False,
                    "n": 1,
                },
                "return_logprob": True,
            },
            None,
        )
    ]
    captures = client.pop_captures("trace-id")
    assert len(captures) == 1
    assert captures[0].sequence_ids == [10, 11, 20, 21]


@pytest.mark.asyncio
async def test_client_applies_miles_max_prompt_len_to_the_initial_render():
    client = MilesSGLangRendererV1Client(
        _args(rollout_max_prompt_len=1),
        config_client=SimpleNamespace(),
        model="test/model",
    )
    client._create_renderer_pool = lambda *_args, **_kwargs: _Renderer()

    with pytest.raises(OverlongPromptError, match="rollout_max_prompt_len=1"):
        await client.get_response(
            ChatDialect(),
            {"messages": [{"role": "user", "content": "question"}]},
            "test/model",
            SamplingConfig(max_tokens=2),
            session_id="trace-id",
        )
