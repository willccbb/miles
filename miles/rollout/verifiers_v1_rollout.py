from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
import uuid
from argparse import Namespace
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from packaging.version import InvalidVersion, Version

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_MIN_VERIFIERS_VERSION = Version("0.2.0")
_MIN_RENDERERS_VERSION = Version("0.1.8")
_GENERATE_PATH = "/generate"
_SGLANG_SAMPLING_KEYS = {
    "custom_params",
    "ebnf",
    "frequency_penalty",
    "ignore_eos",
    "json_schema",
    "logit_bias",
    "max_new_tokens",
    "min_new_tokens",
    "min_p",
    "n",
    "no_stop_trim",
    "presence_penalty",
    "regex",
    "repetition_penalty",
    "sampling_seed",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "stop",
    "stop_regex",
    "stop_token_ids",
    "stream_interval",
    "structural_tag",
    "temperature",
    "top_k",
    "top_p",
}
_V020_SAMPLING_KEYS = {
    "frequency_penalty",
    "logit_bias",
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
    "min_p",
    "n",
    "presence_penalty",
    "repetition_penalty",
    "response_format",
    "seed",
    "stop",
    "stop_sequences",
    "temperature",
    "text",
    "top_k",
    "top_logprobs",
    "top_p",
}


def _load_config_data(path: str) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text()
    if config_path.suffix == ".json":
        data = json.loads(text)
    elif config_path.suffix == ".toml":
        if sys.version_info < (3, 11):
            raise RuntimeError("Verifiers V1 config TOML requires Python 3.11+ for tomllib.")
        import tomllib

        data = tomllib.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the root.")
    return data


def _optional_dependency_error() -> RuntimeError:
    return RuntimeError(
        "Verifiers V1 rollout requires Python 3.11+, verifiers>=0.2.0, and renderers>=0.1.8. "
        "Install with `pip install 'miles[verifiers]'`."
    )


def _check_version(package: str, raw_version: str, minimum: Version) -> None:
    try:
        parsed = Version(raw_version)
    except InvalidVersion as e:
        raise RuntimeError(f"Cannot verify the installed {package} version: {raw_version!r}.") from e
    if parsed < minimum:
        raise RuntimeError(f"Verifiers V1 rollout requires {package}>={minimum}, got {raw_version}.")


def _installed_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError as e:
        raise RuntimeError(
            f"Cannot verify the installed {package} version because its package metadata is missing."
        ) from e


@lru_cache(maxsize=1)
def _import_verifiers_v1():
    if sys.version_info < (3, 11):
        raise _optional_dependency_error()
    try:
        from verifiers.v1 import Environment
        from verifiers.v1.clients import ModelContext
        from verifiers.v1.clients.client import RelayReply
        from verifiers.v1.configs.eval import EvalConfig
        from verifiers.v1.errors import OverlongPromptError, model_error
        from verifiers.v1.types import SamplingConfig
    except ModuleNotFoundError as e:
        raise _optional_dependency_error() from e

    _check_version("verifiers", _installed_version("verifiers"), _MIN_VERIFIERS_VERSION)
    return SimpleNamespace(
        Environment=Environment,
        EvalConfig=EvalConfig,
        ModelContext=ModelContext,
        OverlongPromptError=OverlongPromptError,
        RelayReply=RelayReply,
        SamplingConfig=SamplingConfig,
        model_error=model_error,
    )


@lru_cache(maxsize=1)
def _import_renderer_runtime():
    try:
        from renderers import create_renderer_pool
        from renderers.base import RendererPool, ToolCallParseStatus
        from verifiers.v1.clients.train import response_from_generate, serialize_completion, tool_to_wire
        from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
        from verifiers.v1.dialects.chat import message_to_wire
    except ModuleNotFoundError as e:
        raise _optional_dependency_error() from e

    _check_version("renderers", _installed_version("renderers"), _MIN_RENDERERS_VERSION)
    return SimpleNamespace(
        AnthropicDialect=AnthropicDialect,
        ChatDialect=ChatDialect,
        RendererPool=RendererPool,
        ResponsesDialect=ResponsesDialect,
        ToolCallParseStatus=ToolCallParseStatus,
        create_renderer_pool=create_renderer_pool,
        message_to_wire=message_to_wire,
        response_from_generate=response_from_generate,
        serialize_completion=serialize_completion,
        tool_to_wire=tool_to_wire,
    )


def _generate_url(args: Namespace, model: str) -> str:
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model in routers:
        ip, port = routers[model]
    else:
        ip, port = args.sglang_router_ip, args.sglang_router_port
    return f"http://{ip}:{port}{_GENERATE_PATH}"


async def _sglang_worker_urls(args: Namespace, model: str) -> list[str]:
    from miles.utils.http_utils import get

    router_url = _generate_url(args, model).removesuffix(_GENERATE_PATH)
    if not getattr(args, "use_miles_router", False):
        try:
            response = await get(f"{router_url}/workers")
            return [worker["url"] for worker in response["workers"]]
        except Exception:
            logger.debug("SGLang /workers lookup failed; trying Miles /list_workers.", exc_info=True)
    response = await get(f"{router_url}/list_workers")
    return list(response["urls"])


def _is_valid_incremental_tail(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    roles = [message.get("role") if isinstance(message.get("role"), str) else None for message in messages]
    if roles[-1] == "user":
        return all(role == "tool" for role in roles[:-1])
    return all(role == "tool" for role in roles)


def _has_multimodal_content(messages) -> bool:
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        if any(getattr(part, "type", None) == "image_url" for part in content):
            return True
    return False


def _multimodal_sources(messages) -> dict[str, list[Any]]:
    images = []
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            if getattr(part, "type", None) != "image_url":
                continue
            image_url = getattr(part, "image_url", None)
            source = getattr(image_url, "url", None)
            if source is not None:
                images.append(source)
    return {"images": images} if images else {}


async def _maybe_offload(renderer, fn):
    runtime = _import_renderer_runtime()
    if isinstance(renderer, runtime.RendererPool):
        return await asyncio.to_thread(fn)
    return fn()


def _json_key(value: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(value or {}), sort_keys=True, separators=(",", ":"), default=str)


def _response_format_to_json_schema(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    kind = value.get("type")
    if kind == "json_object":
        return json.dumps({"type": "object"})
    if kind != "json_schema":
        return None
    schema = value.get("json_schema")
    if isinstance(schema, Mapping) and isinstance(schema.get("schema"), Mapping):
        schema = schema["schema"]
    return json.dumps(schema) if isinstance(schema, Mapping) else None


def _parse_request_sampling(dialect, request_body: dict[str, Any]) -> dict[str, Any]:
    """Read effective request sampling on both Verifiers 0.2.0 and newer V1 builds."""
    parse_sampling = getattr(dialect, "parse_sampling", None)
    if parse_sampling is not None:
        return parse_sampling(request_body).model_dump(exclude_none=True)
    return {key: request_body[key] for key in _V020_SAMPLING_KEYS if request_body.get(key) is not None}


def _normalize_sampling_layer(values: Mapping[str, Any]) -> dict[str, Any]:
    sampling = dict(values)
    max_tokens = sampling.pop("max_tokens", None)
    max_tokens = sampling.pop("max_completion_tokens", max_tokens)
    max_tokens = sampling.pop("max_output_tokens", max_tokens)
    if max_tokens is not None:
        sampling["max_new_tokens"] = max_tokens
    if "stop_sequences" in sampling:
        sampling.setdefault("stop", sampling.pop("stop_sequences"))
    if "sampling_seed" not in sampling and "seed" in sampling:
        sampling["sampling_seed"] = sampling.pop("seed")
    else:
        sampling.pop("seed", None)

    schema = _response_format_to_json_schema(sampling.pop("response_format", None))
    text_config = sampling.pop("text", None)
    if schema is None and isinstance(text_config, Mapping):
        schema = _response_format_to_json_schema(text_config.get("format"))
    if schema is not None:
        sampling.setdefault("json_schema", schema)
    return sampling


def _base_sampling_params(dialect, request_body: dict, sampling_args) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_sampling = sampling_args.model_dump(exclude_none=True)
    extra_body = dict(raw_sampling.pop("extra_body", None) or {})
    request_options = {}
    for key in ("chat_template_kwargs", "cache_salt", "priority"):
        default = extra_body.pop(key, None)
        request_options[key] = request_body.get(key, default)
    canonical = _parse_request_sampling(dialect, request_body)
    native = {key: request_body[key] for key in _SGLANG_SAMPLING_KEYS if request_body.get(key) is not None}
    sampling_params = {
        **_normalize_sampling_layer(extra_body),
        **_normalize_sampling_layer({**native, **canonical}),
        # Responses and Anthropic only put provider-native fields on the wire. Keep
        # the full EvalConfig authoritative so SGLang knobs and deterministic seeds
        # survive every renderer dialect.
        **_normalize_sampling_layer(raw_sampling),
    }

    ignored = set(sampling_params) - _SGLANG_SAMPLING_KEYS
    if ignored:
        logger.debug("Ignoring non-SGLang sampling fields from Verifiers request: %s", sorted(ignored))
    sampling_params = {key: value for key, value in sampling_params.items() if key in _SGLANG_SAMPLING_KEYS}
    sampling_params["n"] = 1
    return sampling_params, request_options


def _finalize_sampling_params(
    args: Namespace,
    sampling_params: dict[str, Any],
    stop_token_ids: list[int],
) -> dict[str, Any]:
    sampling_params = dict(sampling_params)
    sampling_params.setdefault("temperature", args.rollout_temperature)
    sampling_params.setdefault("top_p", args.rollout_top_p)
    if args.rollout_top_k is not None:
        sampling_params.setdefault("top_k", args.rollout_top_k)
    sampling_params.setdefault("max_new_tokens", args.rollout_max_response_len)
    if args.rollout_stop is not None:
        sampling_params.setdefault("stop", args.rollout_stop)

    combined_stop_token_ids = list(stop_token_ids)
    if args.rollout_stop_token_ids:
        combined_stop_token_ids.extend(
            token_id for token_id in args.rollout_stop_token_ids if token_id not in combined_stop_token_ids
        )
    if combined_stop_token_ids:
        sampling_params["stop_token_ids"] = combined_stop_token_ids

    sampling_params["skip_special_tokens"] = args.rollout_skip_special_tokens
    sampling_params["no_stop_trim"] = True
    sampling_params["spaces_between_special_tokens"] = False
    return sampling_params


def _sampling_params(args: Namespace, sampling_args, stop_token_ids: list[int]) -> dict[str, Any]:
    raw = sampling_args.model_dump(exclude_none=True)
    extra = dict(raw.pop("extra_body", None) or {})
    extra.pop("chat_template_kwargs", None)
    extra = {**_normalize_sampling_layer(extra), **_normalize_sampling_layer(raw)}
    extra = {key: value for key, value in extra.items() if key in _SGLANG_SAMPLING_KEYS}
    return _finalize_sampling_params(args, extra, stop_token_ids)


def _clamp_max_new_tokens(args: Namespace, prompt_ids: list[int], sampling_params: dict[str, Any]) -> None:
    if args.rollout_max_context_len is None:
        return
    requested = sampling_params.get("max_new_tokens", args.rollout_max_response_len)
    if requested is None:
        return
    sampling_params["max_new_tokens"] = min(requested, args.rollout_max_context_len - len(prompt_ids))


def _opd_requests_student_top_logprobs(args: Namespace) -> bool:
    return (
        getattr(args, "use_opd", False)
        and (getattr(args, "opd_log_prob_top_k", 0) or 0) > 0
        and getattr(args, "opd_top_k_strategy", "only-student") != "only-teacher"
    )


def _build_sglang_generate_payload(
    args: Namespace,
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    multimodal_inputs: Mapping[str, list[Any]] | None = None,
    multi_modal_data: Any = None,
    request_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if is_lora_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME
    if getattr(args, "use_rollout_routing_replay", False):
        payload["return_routed_experts"] = True
        payload["routed_experts_start_len"] = 0
    if getattr(args, "use_rollout_indexer_replay", False):
        payload["return_indexer_topk"] = True
    if _opd_requests_student_top_logprobs(args):
        payload["top_logprobs_num"] = int(args.opd_log_prob_top_k)

    multimodal_inputs = multimodal_inputs or {}
    for source_key, payload_key in (
        ("images", "image_data"),
        ("videos", "video_data"),
        ("audio", "audio_data"),
    ):
        if values := multimodal_inputs.get(source_key):
            payload[payload_key] = list(values)
    if multi_modal_data is not None and getattr(multi_modal_data, "mm_hashes", None):
        if image_hashes := multi_modal_data.mm_hashes.get("image"):
            payload["mm_hashes"] = list(image_hashes)

    request_options = request_options or {}
    if request_options.get("priority") is not None:
        payload["priority"] = request_options["priority"]
    if request_options.get("cache_salt") is not None:
        payload["extra_key"] = request_options["cache_salt"]
    return payload


def _finish_reason(output: dict[str, Any], tool_calls: list[Any], ToolCallParseStatus) -> str | None:
    raw_finish = output.get("meta_info", {}).get("finish_reason")
    if isinstance(raw_finish, dict):
        finish_reason = raw_finish.get("type")
    else:
        finish_reason = raw_finish
    ok_tool_calls = [
        tool_call for tool_call in tool_calls if getattr(tool_call, "status", None) == ToolCallParseStatus.OK
    ]
    if ok_tool_calls and finish_reason == "stop":
        return "tool_calls"
    return finish_reason if finish_reason in {"stop", "length", "tool_calls"} else None


def _decode_replay_array(raw: str, *, rows: int, layers: int, topk: int | None = None) -> np.ndarray:
    values = np.frombuffer(base64.b64decode(raw.encode("ascii")), dtype=np.int32)
    if rows <= 0:
        if len(values):
            raise ValueError("SGLang replay payload has values for an empty token range.")
        width = max(topk or 0, 0)
        return np.empty((0, layers, width), dtype=np.int32)
    if topk is None:
        if layers <= 0 or len(values) % (rows * layers):
            raise ValueError("SGLang replay payload size is not divisible by rows * layers.")
        topk = len(values) // (rows * layers)
    expected = rows * layers * topk
    if len(values) != expected:
        raise ValueError(f"SGLang replay payload has {len(values)} values, expected {expected}.")
    return values.reshape(rows, layers, topk)


def _routed_experts_payload(args: Namespace, meta_info: Mapping[str, Any], sequence_len: int) -> tuple[Any, Any]:
    raw = meta_info.get("routed_experts")
    if raw is None:
        return None, None
    rows = max(sequence_len - 1, 0)
    array = _decode_replay_array(raw, rows=rows, layers=args.num_layers, topk=args.moe_router_topk)
    # Verifiers 0.2.x carries routed experts as uint8 while Miles/SGLang preserve
    # int32. Feed Verifiers its native wire type and retain the lossless array for Miles.
    wire = None
    if array.size == 0 or (array.min() >= 0 and array.max() <= np.iinfo(np.uint8).max):
        wire_array = array.astype(np.uint8, copy=False)
        wire = {
            "data": base64.b64encode(wire_array.tobytes()).decode("ascii"),
            "shape": list(wire_array.shape),
            "start": 0,
        }
    return wire, array


def _indexer_topk_array(meta_info: Mapping[str, Any], sequence_len: int) -> np.ndarray | None:
    raw = meta_info.get("indexer_topk")
    if raw is None:
        return None
    layers = meta_info.get("indexer_topk_num_layers")
    if layers is None:
        raise ValueError("SGLang returned indexer_topk without indexer_topk_num_layers.")
    return _decode_replay_array(raw, rows=max(sequence_len - 1, 0), layers=int(layers))


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    for i in range(limit):
        if left[i] != right[i]:
            return i
    return limit


@dataclass
class _GenerationCapture:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]
    output_text: str
    sampled_mask: list[bool]
    logprobs: list[float]
    top_logprobs: list[list[Any]]
    meta_info: dict[str, Any]
    routed_experts: np.ndarray | None
    indexer_topk: np.ndarray | None
    multimodal_inputs: dict[str, list[Any]]
    multi_modal_data: Any

    @property
    def sequence_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.completion_ids]


def _serialize_responses(response, model: str) -> dict[str, Any]:
    output = []
    if response.message.reasoning_content:
        output.append(
            {
                "id": f"rs_{uuid.uuid4().hex}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": response.message.reasoning_content}],
            }
        )
    if response.message.content:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": response.message.content,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    for call in response.message.tool_calls or []:
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
        )
    usage = response.usage
    return {
        "id": response.id or f"resp_{uuid.uuid4().hex}",
        "created_at": response.created or int(time.time()),
        "model": response.model or model,
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "incomplete" if response.finish_reason == "length" else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if response.finish_reason == "length" else None,
        "usage": (
            {
                "input_tokens": usage.input_tokens,
                "input_tokens_details": {"cached_tokens": usage.cached_input_tokens or 0},
                "output_tokens": usage.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens or 0},
                "total_tokens": usage.total_tokens,
            }
            if usage
            else None
        ),
    }


def _serialize_anthropic(response, model: str) -> dict[str, Any]:
    content = []
    if response.message.reasoning_content:
        content.append(
            {
                "type": "thinking",
                "thinking": response.message.reasoning_content,
                "signature": "",
            }
        )
    if response.message.content:
        content.append({"type": "text", "text": response.message.content})
    for call in response.message.tool_calls or []:
        try:
            tool_input = json.loads(call.arguments)
        except (TypeError, ValueError):
            tool_input = {"raw": call.arguments}
        content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": tool_input})
    usage = response.usage
    stop_reason = {
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "stop": "end_turn",
    }.get(response.finish_reason, "end_turn")
    return {
        "id": response.id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": response.model or model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.input_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": (usage.cached_input_tokens or 0) if usage else 0,
        },
    }


def _serialize_completion(response, dialect, model: str) -> dict[str, Any]:
    runtime = _import_renderer_runtime()
    if isinstance(dialect, runtime.ChatDialect):
        return runtime.serialize_completion(response, model)
    if isinstance(dialect, runtime.ResponsesDialect):
        return _serialize_responses(response, model)
    if isinstance(dialect, runtime.AnthropicDialect):
        return _serialize_anthropic(response, model)
    raise NotImplementedError(f"No renderer response serializer for {type(dialect).__name__}.")


def _sse(data: Mapping[str, Any] | str) -> bytes:
    encoded = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    return f"data: {encoded}\n\n".encode()


def _stream_chunks(dialect, raw: dict[str, Any]) -> list[bytes]:
    runtime = _import_renderer_runtime()
    if isinstance(dialect, runtime.ChatDialect):
        choice = raw["choices"][0]
        delta = dict(choice["message"])
        if delta.get("tool_calls"):
            delta["tool_calls"] = [
                {**tool_call, "index": index} for index, tool_call in enumerate(delta["tool_calls"])
            ]
        return [
            _sse(
                {
                    "id": raw["id"],
                    "object": "chat.completion.chunk",
                    "created": raw["created"],
                    "model": raw["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": choice["finish_reason"],
                        }
                    ],
                    "usage": raw.get("usage"),
                }
            ),
            _sse("[DONE]"),
        ]
    if isinstance(dialect, runtime.ResponsesDialect):
        event_type = "response.incomplete" if raw.get("status") == "incomplete" else "response.completed"
        return [
            _sse({"type": event_type, "response": raw, "sequence_number": 0}),
            _sse("[DONE]"),
        ]
    if isinstance(dialect, runtime.AnthropicDialect):
        chunks = [
            _sse(
                {
                    "type": "message_start",
                    "message": {**raw, "content": [], "stop_reason": None, "stop_sequence": None},
                }
            )
        ]
        for index, block in enumerate(raw["content"]):
            chunks.append(_sse({"type": "content_block_start", "index": index, "content_block": block}))
            chunks.append(_sse({"type": "content_block_stop", "index": index}))
        chunks.extend(
            [
                _sse(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": raw["stop_reason"], "stop_sequence": raw["stop_sequence"]},
                        "usage": {"output_tokens": raw["usage"]["output_tokens"]},
                    }
                ),
                _sse({"type": "message_stop"}),
            ]
        )
        return chunks
    raise NotImplementedError(f"No renderer stream serializer for {type(dialect).__name__}.")


class MilesSGLangRendererV1Client:
    """Verifiers V1 renderer client backed by Miles' active SGLang engine."""

    def __init__(
        self,
        args: Namespace,
        *,
        config_client: Any,
        model: str,
        renderer_cache: dict[str, Any] | None = None,
    ):
        runtime = _import_renderer_runtime()
        self.args = args
        self.model = model
        self._create_renderer_pool = runtime.create_renderer_pool
        self.renderer_config = getattr(config_client, "renderer", None)
        self.renderer_model_name = getattr(config_client, "renderer_model_name", None) or args.hf_checkpoint or model
        self.pool_size = getattr(config_client, "pool_size", 1) or 1
        self._renderers = renderer_cache if renderer_cache is not None else {}
        self._captures: dict[str, list[_GenerationCapture]] = {}

    def _renderer_pool(self, chat_template_kwargs: Mapping[str, Any] | None):
        key = _json_key(chat_template_kwargs)
        if key not in self._renderers:
            pool_kwargs: dict[str, Any] = {"size": self.pool_size}
            if chat_template_kwargs:
                pool_kwargs["chat_template_kwargs"] = dict(chat_template_kwargs)
            self._renderers[key] = self._create_renderer_pool(
                self.renderer_model_name,
                self.renderer_config,
                **pool_kwargs,
            )
        return self._renderers[key]

    def _prompt_history(
        self,
        session_id: str | None,
        prompt_ids: list[int],
    ) -> tuple[list[bool], list[float], list[list[Any]]]:
        prompt_mask = [False] * len(prompt_ids)
        prompt_logprobs = [0.0] * len(prompt_ids)
        prompt_top: list[list[Any]] = [[] for _ in prompt_ids]
        if session_id is None:
            return prompt_mask, prompt_logprobs, prompt_top
        best_capture = None
        best_prefix = 0
        for capture in self._captures.get(session_id, []):
            prefix = _longest_common_prefix(capture.sequence_ids, prompt_ids)
            if prefix > best_prefix:
                best_capture, best_prefix = capture, prefix
        if best_capture is not None:
            prompt_mask[:best_prefix] = best_capture.sampled_mask[:best_prefix]
            prompt_logprobs[:best_prefix] = best_capture.logprobs[:best_prefix]
            prompt_top[:best_prefix] = best_capture.top_logprobs[:best_prefix]
        return prompt_mask, prompt_logprobs, prompt_top

    def _record_capture(
        self,
        session_id: str | None,
        *,
        prompt_ids: list[int],
        completion_ids: list[int],
        completion_logprobs: list[float],
        output_text: str,
        output_top_logprobs: list[list[Any]],
        meta_info: dict[str, Any],
        routed_experts: np.ndarray | None,
        indexer_topk: np.ndarray | None,
        multimodal_inputs: dict[str, list[Any]],
        multi_modal_data: Any,
    ) -> None:
        if session_id is None:
            return
        prompt_mask, prompt_logprobs, prompt_top = self._prompt_history(session_id, prompt_ids)
        capture = _GenerationCapture(
            prompt_ids=list(prompt_ids),
            completion_ids=list(completion_ids),
            completion_logprobs=list(completion_logprobs),
            output_text=output_text,
            sampled_mask=[*prompt_mask, *([True] * len(completion_ids))],
            logprobs=[*prompt_logprobs, *completion_logprobs],
            top_logprobs=[*prompt_top, *output_top_logprobs],
            meta_info=meta_info,
            routed_experts=routed_experts,
            indexer_topk=indexer_topk,
            multimodal_inputs=multimodal_inputs,
            multi_modal_data=multi_modal_data,
        )
        self._captures.setdefault(session_id, []).append(capture)

    def pop_captures(self, trace_id: str) -> list[_GenerationCapture]:
        return self._captures.pop(trace_id, [])

    async def _generate_response(
        self,
        dialect,
        body: dict,
        model: str,
        sampling_args,
        session_id: str | None,
        turn,
    ):
        renderer_runtime = _import_renderer_runtime()
        verifiers_runtime = _import_verifiers_v1()
        request_body = dialect.apply_overrides(body, model, sampling_args)
        parsed_prompt, tools = dialect.parse_request(request_body)
        prompt = turn.prompt if turn is not None else parsed_prompt
        base_sampling_params, request_options = _base_sampling_params(dialect, request_body, sampling_args)
        renderer = self._renderer_pool(request_options["chat_template_kwargs"])
        wire_tools = [renderer_runtime.tool_to_wire(tool) for tool in tools] if tools else None

        prompt_ids: list[int] | None = None
        multi_modal_data = None
        prompt_attribution = None
        bridged_turn = None
        if turn is not None:
            tail_messages = [renderer_runtime.message_to_wire(message) for message in turn.tail]
            can_bridge = not _has_multimodal_content(prompt) and _is_valid_incremental_tail(tail_messages)
            previous_ids = turn.previous_token_ids() if can_bridge else None
            if previous_ids is not None:
                previous_prompt_ids, previous_completion_ids = previous_ids

                def bridge():
                    return renderer.bridge_to_next_turn(
                        previous_prompt_ids,
                        previous_completion_ids,
                        tail_messages,
                        tools=wire_tools,
                    )

                bridged = await _maybe_offload(renderer, bridge)
                if bridged is not None:
                    prompt_ids = list(bridged.token_ids)
                    multi_modal_data = bridged.multi_modal_data
                    prompt_attribution = bridged
                    bridged_turn = turn

        if prompt_ids is None:
            wire_messages = [renderer_runtime.message_to_wire(message) for message in prompt]

            def render():
                return renderer.render(wire_messages, tools=wire_tools, add_generation_prompt=True)

            rendered = await _maybe_offload(renderer, render)
            prompt_ids = list(rendered.token_ids)
            multi_modal_data = rendered.multi_modal_data
            prompt_attribution = rendered

        max_prompt_len = getattr(self.args, "rollout_max_prompt_len", None)
        if (
            max_prompt_len is not None
            and not self._captures.get(session_id or "")
            and len(prompt_ids) > max_prompt_len
        ):
            raise verifiers_runtime.OverlongPromptError(
                f"initial prompt has {len(prompt_ids)} tokens, rollout_max_prompt_len={max_prompt_len}"
            )

        sampling_params = _finalize_sampling_params(self.args, base_sampling_params, renderer.get_stop_token_ids())
        _clamp_max_new_tokens(self.args, prompt_ids, sampling_params)
        if sampling_params.get("max_new_tokens", 1) <= 0:
            raise verifiers_runtime.OverlongPromptError(
                f"prompt has {len(prompt_ids)} tokens, rollout_max_context_len={self.args.rollout_max_context_len}"
            )

        multimodal_inputs = _multimodal_sources(prompt)
        payload = _build_sglang_generate_payload(
            self.args,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            multimodal_inputs=multimodal_inputs,
            multi_modal_data=multi_modal_data,
            request_options=request_options,
        )
        request_headers = None
        if getattr(self.args, "sglang_router_policy", None) == "consistent_hashing" and session_id is not None:
            request_headers = {"X-SMG-Routing-Key": session_id}
        try:
            from miles.utils.http_utils import post

            output = await post(_generate_url(self.args, model), payload, headers=request_headers)
        except Exception as e:
            raise verifiers_runtime.model_error(str(e)) from e

        meta_info = dict(output.get("meta_info") or {})
        output_token_logprobs = meta_info.get("output_token_logprobs") or []
        completion_ids = [int(item[1]) for item in output_token_logprobs]
        completion_logprobs = [float(item[0]) for item in output_token_logprobs]
        expected_completion_tokens = meta_info.get("completion_tokens", len(completion_ids))
        if len(completion_ids) != expected_completion_tokens:
            raise verifiers_runtime.model_error(
                "SGLang generate response has mismatched completion token metadata: "
                f"{len(completion_ids)} != {expected_completion_tokens}"
            )
        output_top_logprobs = list(meta_info.get("output_top_logprobs") or [])
        if output_top_logprobs and len(output_top_logprobs) != len(completion_ids):
            raise verifiers_runtime.model_error(
                "SGLang generate response has mismatched output_top_logprobs: "
                f"{len(output_top_logprobs)} != {len(completion_ids)}"
            )
        if not output_top_logprobs:
            output_top_logprobs = [[] for _ in completion_ids]

        sequence_len = len(prompt_ids) + len(completion_ids)
        try:
            routed_wire, routed_array = _routed_experts_payload(self.args, meta_info, sequence_len)
            indexer_array = _indexer_topk_array(meta_info, sequence_len)
        except ValueError as e:
            raise verifiers_runtime.model_error(str(e)) from e
        if getattr(self.args, "use_rollout_routing_replay", False) and routed_array is None:
            raise verifiers_runtime.model_error("SGLang did not return routed_experts requested by Miles.")
        if getattr(self.args, "use_rollout_indexer_replay", False) and indexer_array is None:
            raise verifiers_runtime.model_error("SGLang did not return indexer_topk requested by Miles.")
        if _opd_requests_student_top_logprobs(self.args) and completion_ids and not any(output_top_logprobs):
            raise verifiers_runtime.model_error("SGLang did not return output_top_logprobs requested by Miles OPD.")

        parsed = await _maybe_offload(
            renderer,
            lambda: renderer.parse_response(completion_ids, tools=wire_tools),
        )
        result = {
            "request_id": output.get("request_id") or f"vf-{uuid.uuid4().hex}",
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "completion_logprobs": completion_logprobs,
            "content": parsed.content,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": parsed.tool_calls,
            "finish_reason": _finish_reason(output, parsed.tool_calls, renderer_runtime.ToolCallParseStatus),
            "routed_experts": routed_wire,
            "multi_modal_data": multi_modal_data,
            "prompt_attribution": prompt_attribution,
        }
        response = renderer_runtime.response_from_generate(result, model, bridged_turn)
        response.raw = _serialize_completion(response, dialect, model)
        self._record_capture(
            session_id,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=completion_logprobs,
            output_text=str(output.get("text") or ""),
            output_top_logprobs=output_top_logprobs,
            meta_info=meta_info,
            routed_experts=routed_array,
            indexer_topk=indexer_array,
            multimodal_inputs=multimodal_inputs,
            multi_modal_data=multi_modal_data,
        )
        return response

    async def get_response(
        self,
        dialect,
        body: dict,
        model: str,
        sampling_args,
        session_id: str | None = None,
        turn=None,
        headers: Mapping[str, str] | None = None,
    ):
        del headers
        return await self._generate_response(dialect, body, model, sampling_args, session_id, turn)

    async def relay(
        self,
        dialect,
        body: dict,
        model: str,
        sampling_args,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        del headers
        response = await self._generate_response(dialect, body, model, sampling_args, session_id, None)
        chunks = _stream_chunks(dialect, response.raw)

        async def iterate() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        async def close() -> None:
            return None

        return _import_verifiers_v1().RelayReply(
            content_type="text/event-stream",
            chunks=iterate(),
            close=close,
        )

    async def relay_aux(
        self,
        dialect,
        route: str,
        body: dict,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, int]:
        del headers
        if route.rstrip("/") != "/v1/messages/count_tokens":
            raise NotImplementedError(f"Unsupported Verifiers auxiliary route: {route}")
        prompt, tools = dialect.parse_request(body)
        runtime = _import_renderer_runtime()
        renderer = self._renderer_pool(None)
        wire_messages = [runtime.message_to_wire(message) for message in prompt]
        wire_tools = [runtime.tool_to_wire(tool) for tool in tools] if tools else None
        rendered = await _maybe_offload(
            renderer,
            lambda: renderer.render(wire_messages, tools=wire_tools, add_generation_prompt=True),
        )
        return {"input_tokens": len(rendered.token_ids)}

    async def close(self) -> None:
        for renderer in self._renderers.values():
            close = getattr(renderer, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result


def _multimodal_train_inputs(multi_modal_data: Any) -> dict[str, torch.Tensor] | None:
    if multi_modal_data is None or multi_modal_data.is_empty():
        return None
    values_by_key: dict[str, list[torch.Tensor]] = {}
    for items in multi_modal_data.mm_items.values():
        for item in items:
            for key, value in item.items():
                tensor = torch.as_tensor(value)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                values_by_key.setdefault(key, []).append(tensor)
    return {key: torch.cat(values, dim=0) for key, values in values_by_key.items()} or None


def _message_text(message: Any) -> str:
    parts = []
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning:
        parts.append(reasoning)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        parts.extend(text for part in content if isinstance((text := getattr(part, "text", None)), str))
    return "".join(parts)


def _branch_response(
    branch,
    fallback: str,
    captures: list[_GenerationCapture],
) -> str:
    """Mirror Miles multi-turn samples: generated turns plus intervening observations."""
    started = False
    parts = []
    sampled_count = sum(bool(getattr(node, "sampled", False)) for node in branch.nodes)
    aligned_captures: list[_GenerationCapture | None] = [None] * max(sampled_count - len(captures), 0)
    if sampled_count:
        aligned_captures.extend(captures[-sampled_count:])
    capture_iter = iter(aligned_captures)
    for node in branch.nodes:
        sampled = bool(getattr(node, "sampled", False))
        started = started or sampled
        if started:
            capture = next(capture_iter, None) if sampled else None
            parts.append(capture.output_text if capture and capture.output_text else _message_text(node.message))
    return "".join(parts).strip() or fallback


def _branch_captures(
    branch, captures: list[_GenerationCapture]
) -> tuple[_GenerationCapture | None, list[_GenerationCapture]]:
    tokens = list(branch.token_ids)
    matching_prefixes = [
        capture
        for capture in captures
        if len(capture.sequence_ids) <= len(tokens) and tokens[: len(capture.sequence_ids)] == capture.sequence_ids
    ]
    exact = next(
        (capture for capture in reversed(matching_prefixes) if capture.sequence_ids == tokens),
        None,
    )
    return exact, matching_prefixes


def _sample_status(trace) -> Sample.Status:
    if trace.has_error:
        return Sample.Status.FAILED
    if trace.is_truncated:
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _trace_branch_to_sample(
    args: Namespace,
    trace,
    branch,
    *,
    captures: list[_GenerationCapture],
    group_index: int,
    index: int,
    allow_capture_without_branch_tokens: bool = False,
) -> Sample:
    capture, branch_captures = _branch_captures(branch, captures)
    if capture is None and allow_capture_without_branch_tokens and not branch.token_ids and captures:
        # Verifiers' streamed relay parser commits the native response message but
        # intentionally has no provider token sidecar. Miles generated the response,
        # so its final capture is the authoritative full-sequence training record.
        capture = captures[-1]
        branch_captures = captures

    if capture is not None:
        tokens = list(capture.sequence_ids)
        mask = list(capture.sampled_mask)
        logprobs = list(capture.logprobs)
    else:
        tokens = list(branch.token_ids)
        mask = list(branch.sampled_mask)
        logprobs = list(branch.logprobs)
    if len(tokens) != len(mask):
        raise ValueError(f"Trace {trace.id} token/mask length mismatch: {len(tokens)} != {len(mask)}")
    first = mask.index(True) if True in mask else len(tokens)
    response_length = len(tokens) - first
    loss_mask = [int(value) for value in mask[first:]]
    rollout_log_probs = logprobs[first:]
    if len(rollout_log_probs) != response_length:
        raise ValueError(f"Trace {trace.id} logprob length mismatch: {len(rollout_log_probs)} != {response_length}")
    if getattr(args, "use_rollout_routing_replay", False) and capture is None:
        raise ValueError(f"Trace {trace.id} has no exact Miles capture for routing replay.")
    if getattr(args, "use_rollout_indexer_replay", False) and capture is None:
        raise ValueError(f"Trace {trace.id} has no exact Miles capture for indexer replay.")
    if _opd_requests_student_top_logprobs(args) and capture is None:
        raise ValueError(f"Trace {trace.id} has no exact Miles capture for OPD top logprobs.")

    reward = trace.reward
    if args.reward_key is not None:
        reward = {**trace.rewards, "reward": trace.reward}
    task_data = trace.task.data
    metadata = {
        "verifiers_v1_trace_id": trace.id,
        "verifiers_v1_branch_index": branch.index,
        "verifiers_v1_task_idx": getattr(task_data, "idx", None),
        "verifiers_v1_rewards": dict(trace.rewards),
        "verifiers_v1_metrics": dict(trace.metrics),
        "verifiers_v1_stop_condition": trace.stop_condition,
    }
    if trace.error is not None:
        metadata["verifiers_v1_error"] = trace.error.model_dump(mode="json", exclude_none=True)
    if capture is not None and _opd_requests_student_top_logprobs(args):
        metadata["opd_student_top_logprobs"] = capture.top_logprobs[first:]

    branch_multimodal_data = getattr(branch, "multi_modal_data", None)
    multi_modal_data = capture.multi_modal_data if capture is not None else branch_multimodal_data
    multimodal_inputs = (
        capture.multimodal_inputs
        if capture is not None
        else _multimodal_sources([node.message for node in branch.nodes])
    )

    label = getattr(task_data, "label", None)
    if label is None:
        label = getattr(task_data, "answer", None)
    sample = Sample(
        group_index=group_index,
        index=index,
        prompt=getattr(task_data, "prompt", "") or "",
        tokens=tokens,
        multimodal_inputs=multimodal_inputs or None,
        multimodal_train_inputs=_multimodal_train_inputs(multi_modal_data),
        response=_branch_response(branch, trace.last_reply, branch_captures),
        response_length=response_length,
        label=label,
        reward=reward,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        rollout_routed_experts=capture.routed_experts if capture is not None else None,
        rollout_indexer_topk=capture.indexer_topk if capture is not None else None,
        status=_sample_status(trace),
        metadata=metadata,
        session_id=trace.id,
    )
    for branch_capture in branch_captures:
        meta_info = branch_capture.meta_info
        if getattr(args, "sglang_speculative_algorithm", None):
            sample.spec_info.add(meta_info)
        sample.prefix_cache_info.add(meta_info)
        if "weight_version" in meta_info:
            sample.weight_versions.append(str(meta_info["weight_version"]))
    timing = getattr(getattr(trace, "timing", None), "generation", None)
    harness_timing = getattr(timing, "harness", None)
    sample.non_generation_time = float(getattr(harness_timing, "duration", 0.0) or 0.0)
    sample.validate()
    return sample


def trace_to_samples(
    args: Namespace,
    trace,
    *,
    captures: list[_GenerationCapture] | None = None,
    group_index: int,
    index_start: int,
) -> list[Sample]:
    branches = trace.branches
    if not branches:
        logger.warning("Verifiers trace %s has no graph branches; omitting it from training.", trace.id)
        return []
    return [
        _trace_branch_to_sample(
            args,
            trace,
            branch,
            captures=captures or [],
            group_index=group_index,
            index=index_start + branch_offset,
            allow_capture_without_branch_tokens=len(branches) == 1,
        )
        for branch_offset, branch in enumerate(branches)
    ]


def trace_to_sample(args: Namespace, trace, *, group_index: int, index: int) -> Sample:
    """Compatibility helper for callers that expect a single-branch trace."""
    samples = trace_to_samples(args, trace, group_index=group_index, index_start=index)
    if len(samples) != 1:
        raise ValueError(f"Verifiers V1 trace {trace.id} produced {len(samples)} branches, expected one.")
    return samples[0]


def _trace_metrics(traces) -> dict[str, float]:
    if not traces:
        return {}
    rewards = [trace.reward for trace in traces]
    return {
        "verifiers_v1/reward_mean": sum(rewards) / len(rewards),
        "verifiers_v1/error_rate": sum(1 for trace in traces if trace.has_error) / len(traces),
        "verifiers_v1/truncated_rate": sum(1 for trace in traces if trace.is_truncated) / len(traces),
        "verifiers_v1/num_turns_mean": sum(trace.num_turns for trace in traces) / len(traces),
        "verifiers_v1/num_branches_mean": sum(len(trace.branches) for trace in traces) / len(traces),
    }


def _flatten_samples(values: Iterable[Any]) -> list[Sample]:
    flattened = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten_samples(value))
        else:
            flattened.append(value)
    return flattened


class VerifiersV1RolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        runtime = _import_verifiers_v1()
        self.args = input.args
        self.data_source = input.data_source
        self.config = runtime.EvalConfig.model_validate(_load_config_data(self.args.verifiers_v1_config))
        if self.config.is_legacy:
            raise ValueError("Miles' Verifiers integration is V1 only; configure a V1 taskset.")
        self.env = runtime.Environment(self.config)
        self.model = self.args.verifiers_v1_model or self.args.hf_checkpoint or self.config.model
        self.sampling = self._sampling_config(runtime.SamplingConfig, self.args)
        self.eval_args = Namespace(**vars(self.args))
        for eval_name, rollout_name in (
            ("eval_temperature", "rollout_temperature"),
            ("eval_top_p", "rollout_top_p"),
            ("eval_top_k", "rollout_top_k"),
            ("eval_max_response_len", "rollout_max_response_len"),
            ("eval_max_prompt_len", "rollout_max_prompt_len"),
            ("eval_max_context_len", "rollout_max_context_len"),
        ):
            value = getattr(self.args, eval_name, None)
            if value is not None:
                setattr(self.eval_args, rollout_name, value)
        self.eval_sampling = self._sampling_config(
            runtime.SamplingConfig,
            self.eval_args,
            min_new_tokens=getattr(self.args, "eval_min_new_tokens", None),
        )
        engine_concurrency = (
            self.args.sglang_server_concurrency * self.args.rollout_num_gpus // self.args.rollout_num_gpus_per_engine
        )
        configured_concurrency = self.config.max_concurrent or engine_concurrency
        self.max_concurrent = self.args.verifiers_v1_max_concurrent or min(
            configured_concurrency,
            engine_concurrency,
        )
        self.client = MilesSGLangRendererV1Client(self.args, config_client=self.config.client, model=self.model)
        self.ctx = runtime.ModelContext(client=self.client, model=self.model, sampling=self.sampling)
        self.eval_client = MilesSGLangRendererV1Client(
            self.eval_args,
            config_client=self.config.client,
            model=self.model,
            renderer_cache=self.client._renderers,
        )
        self.eval_ctx = runtime.ModelContext(client=self.eval_client, model=self.model, sampling=self.eval_sampling)
        from miles.utils.misc import load_function

        self.dynamic_filter = (
            load_function(self.args.dynamic_sampling_filter_path)
            if self.args.dynamic_sampling_filter_path is not None
            else None
        )
        self._infinite = bool(type(self.env.taskset).INFINITE)
        self._task_iter = iter(self.env.taskset.load()) if self._infinite else None
        self._tasks = [] if self._infinite else self.env.taskset.select(self.config.num_tasks, self.config.shuffle)
        if not self._infinite and not self._tasks:
            raise ValueError("Verifiers V1 taskset selected zero tasks.")
        self._next_train_task_idx = self.args.verifiers_v1_task_offset
        self._next_group_index = 0
        self._next_sample_index = 0

    def _sampling_config(self, SamplingConfig, args: Namespace, *, min_new_tokens: int | None = None):
        data = self.config.sampling.model_dump(exclude_none=True)
        data.setdefault("temperature", args.rollout_temperature)
        data.setdefault("top_p", args.rollout_top_p)
        if args.rollout_top_k is not None:
            data.setdefault("top_k", args.rollout_top_k)
        data.setdefault("max_tokens", args.rollout_max_response_len)
        extra_body = data.get("extra_body") or {}
        if min_new_tokens is not None and "min_new_tokens" not in data and "min_new_tokens" not in extra_body:
            data["min_new_tokens"] = min_new_tokens
        return SamplingConfig.model_validate(data)

    def _task(self, idx: int):
        if self._infinite:
            assert self._task_iter is not None
            while len(self._tasks) <= idx:
                self._tasks.append(next(self._task_iter))
            return self._tasks[idx]
        return self._tasks[idx % len(self._tasks)]

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        return await self._call_train(input)

    async def _run_task_group(
        self,
        task,
        n: int,
        semaphore: asyncio.Semaphore,
        seed_base: int,
        ctx=None,
    ):
        ctx = ctx or self.ctx
        episode = self.env.episode(task, ctx, n=n)
        if getattr(self.args, "sglang_enable_deterministic_inference", False):
            runtime = _import_verifiers_v1()
            for offset, rollout in enumerate(episode.rollouts):
                seeded = ctx.sampling.model_copy(update={"sampling_seed": seed_base + offset})
                rollout.ctx = runtime.ModelContext(client=ctx.client, model=self.model, sampling=seeded)
        return await episode.run(semaphore)

    def _convert_group(
        self,
        traces,
        *,
        group_index: int,
        preserve_empty: bool = False,
        client: MilesSGLangRendererV1Client | None = None,
    ) -> list[Sample | list[Sample] | None]:
        client = client or self.client
        group = []
        complete = True
        for trace in traces:
            converted = trace_to_samples(
                self.args,
                trace,
                captures=client.pop_captures(trace.id),
                group_index=group_index,
                index_start=self._next_sample_index,
            )
            self._next_sample_index += len(converted)
            if not converted:
                complete = False
                if preserve_empty:
                    group.append(None)
                continue
            group.append(converted[0] if len(converted) == 1 else converted)
        return group if complete or preserve_empty else []

    async def _apply_miles_rewards(self, group: list[Sample | list[Sample]]) -> None:
        from miles.rollout.rm_hub import async_rm, batched_async_rm

        samples = _flatten_samples(group)
        if not samples:
            return
        if self.args.group_rm:
            rewards = await batched_async_rm(self.args, samples)
        elif self.args.custom_rm_path is not None or self.args.rm_type:
            rewards = await asyncio.gather(*(async_rm(self.args, sample) for sample in samples))
        else:
            return
        if rewards is None or len(rewards) != len(samples):
            raise ValueError(
                f"Miles reward model returned {0 if rewards is None else len(rewards)} rewards "
                f"for {len(samples)} Verifiers samples."
            )
        for sample, reward in zip(samples, rewards, strict=True):
            sample.reward = reward

    async def _postprocess_train_samples(self, data, all_data) -> None:
        from miles.utils.misc import load_function

        if function := load_function(self.args.rollout_sample_filter_path):
            function(self.args, data)
        if function := load_function(self.args.rollout_all_samples_process_path):
            function(self.args, all_data, self.data_source)
        await recompute_samples_rollout_logprobs_via_prefill(
            self.args,
            _flatten_samples(data),
            url=_generate_url(self.args, self.model),
            sampling_params=_sampling_params(self.args, self.sampling, []),
        )

    async def _cancel_pending(self, futures: Iterable[asyncio.Task]) -> None:
        pending = [future for future in futures if not future.done()]
        for future in pending:
            future.cancel()
        if pending:
            try:
                from miles.utils.http_utils import post

                urls = await _sglang_worker_urls(self.args, self.model)
                results = await asyncio.gather(
                    *(post(f"{url}/abort_request", {"abort_all": True}) for url in urls),
                    return_exceptions=True,
                )
                for url, result in zip(urls, results, strict=True):
                    if isinstance(result, Exception):
                        logger.warning("Failed to abort pending Verifiers requests at %s: %s", url, result)
            except Exception:
                logger.exception("Failed to enumerate SGLang workers while canceling Verifiers episodes.")
        await asyncio.gather(*futures, return_exceptions=True)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        from miles.utils import dumper_utils

        await dumper_utils.configure_sglang(self.args)
        target = self.args.rollout_batch_size
        groups: list[list[Sample | list[Sample]]] = []
        all_groups: list[list[Sample | list[Sample]]] = []
        all_traces = []
        metrics = MetricGatherer()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        pending: set[asyncio.Task] = set()

        async with self.env.serving():
            try:
                while len(groups) < target:
                    while len(groups) + len(pending) < target:
                        for _ in range(self.args.over_sampling_batch_size):
                            task_idx = self._next_train_task_idx
                            self._next_train_task_idx += 1
                            task = self._task(task_idx)
                            seed_base = self.args.rollout_seed + task_idx * self.args.n_samples_per_prompt
                            pending.add(
                                asyncio.create_task(
                                    self._run_task_group(
                                        task,
                                        self.args.n_samples_per_prompt,
                                        semaphore,
                                        seed_base,
                                    )
                                )
                            )
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task_future in done:
                        try:
                            traces = task_future.result()
                        except Exception:
                            logger.exception("Verifiers V1 task group failed before producing traces; resampling.")
                            metrics.on_dynamic_filter_drop(reason="episode_error")
                            continue
                        all_traces.extend(traces)
                        group_index = self._next_group_index
                        self._next_group_index += 1
                        group = self._convert_group(
                            traces,
                            group_index=group_index,
                        )
                        if len(group) != self.args.n_samples_per_prompt:
                            metrics.on_dynamic_filter_drop(reason="empty_trace")
                            continue
                        await self._apply_miles_rewards(group)
                        all_groups.append(group)
                        dynamic_filter_output = call_dynamic_filter(self.dynamic_filter, self.args, group)
                        if not dynamic_filter_output.keep:
                            metrics.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                            continue
                        if len(groups) < target:
                            groups.append(group)
            finally:
                await self._cancel_pending(pending)

        groups.sort(key=lambda group: _flatten_samples(group)[0].index)
        all_groups.sort(key=lambda group: _flatten_samples(group)[0].index)
        await self._postprocess_train_samples(groups, all_groups)
        collected_metrics = metrics.collect()
        collected_metrics.update(_trace_metrics(all_traces))
        return RolloutFnTrainOutput(samples=groups, metrics=collected_metrics)

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        from miles.utils import dumper_utils

        await dumper_utils.configure_sglang(self.args)
        num_tasks = self.args.verifiers_v1_num_eval_tasks or self.args.rollout_batch_size
        semaphore = asyncio.Semaphore(self.max_concurrent)
        futures = []
        async with self.env.serving():
            try:
                for i in range(num_tasks):
                    seed_base = self.args.rollout_seed + i * self.args.n_samples_per_eval_prompt
                    futures.append(
                        asyncio.create_task(
                            self._run_task_group(
                                self._task(self.args.verifiers_v1_task_offset + i),
                                self.args.n_samples_per_eval_prompt,
                                semaphore,
                                seed_base,
                                self.eval_ctx,
                            )
                        )
                    )
                trace_groups = await asyncio.gather(*futures)
            finally:
                await self._cancel_pending(futures)

        eval_samples = []
        rewards = []
        truncated = []
        all_traces = []
        uses_miles_reward_model = bool(self.args.group_rm or self.args.custom_rm_path is not None or self.args.rm_type)
        reward_key = self.args.eval_reward_key or self.args.reward_key
        for i, traces in enumerate(trace_groups):
            all_traces.extend(traces)
            aligned_group = self._convert_group(
                traces,
                group_index=i,
                preserve_empty=True,
                client=self.eval_client,
            )
            trainable_group = [value for value in aligned_group if value is not None]
            await self._apply_miles_rewards(trainable_group)
            eval_samples.extend(_flatten_samples(trainable_group))
            for trace, value in zip(traces, aligned_group, strict=True):
                if uses_miles_reward_model:
                    branch_rewards = (
                        [
                            sample.reward if reward_key is None else sample.reward[reward_key]
                            for sample in _flatten_samples([value])
                        ]
                        if value is not None
                        else []
                    )
                    reward = (
                        sum(branch_rewards) / len(branch_rewards)
                        if branch_rewards and all(item is not None for item in branch_rewards)
                        else None
                    )
                elif reward_key is None:
                    reward = trace.reward
                else:
                    trace_rewards = {**trace.rewards, "reward": trace.reward}
                    reward = trace_rewards[reward_key]
                rewards.append(reward)
                truncated.append(trace.is_truncated)
        return RolloutFnEvalOutput(
            data={
                self.config.env_id
                or "verifiers_v1": {
                    "rewards": rewards,
                    "truncated": truncated,
                    "samples": eval_samples,
                }
            },
            metrics=_trace_metrics(all_traces),
        )


_LEGACY_INSTANCES: dict[tuple[int, int], VerifiersV1RolloutFn] = {}


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Legacy and refactored Miles entrypoint backed by one persistent V1 adapter."""
    from miles.utils.async_utils import run

    key = (id(args), id(data_source))
    adapter = _LEGACY_INSTANCES.get(key)
    if adapter is None:
        adapter = VerifiersV1RolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
        _LEGACY_INSTANCES[key] = adapter
    input = RolloutFnEvalInput(rollout_id) if evaluation else RolloutFnTrainInput(rollout_id)
    return run(adapter(input))
