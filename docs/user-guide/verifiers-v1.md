---
title: Verifiers V1 Rollout
description: Use Verifiers V1 tasksets and harnesses as a Miles rollout source.
---

Miles can run a Verifiers V1 environment in place of its prompt dataset while keeping the
SGLang engine and the rest of the Miles training pipeline. This integration requires Python
3.11 or newer and supports `verifiers>=0.2.0` with `renderers>=0.1.8`.

## Install

Install the optional dependencies with the project:

```bash
pip install -e '.[verifiers]'
```

## Configure Verifiers

Create a V1 EvalConfig-compatible TOML, JSON, or YAML file. For example:

```toml
num_tasks = 128
shuffle = true

[taskset]
id = "gsm8k-v1"

[client]
type = "train"
```

The taskset, task arguments, harness, runtime, limits, retries, judges, and scoring rules come
from this file. Miles embeds the environment, so the Verifiers CLI presentation and publishing
fields (`rich`, `server`, `push`, and `output_dir`) are not used.

## Launch

Add these flags to the normal Miles training command:

```bash
--use-verifiers-v1 \
--verifiers-v1-config /path/to/verifiers.toml
```

`--verifiers-v1-model` changes the model name exposed to the harness and defaults to
`--hf-checkpoint`. `--verifiers-v1-task-offset` selects the first task. Training cycles a finite
selected taskset and advances the task cursor across rollout batches.

Miles owns rollout group sizes and concurrency:

- `--n-samples-per-prompt` replaces the config's `num_rollouts` for training.
- `--n-samples-per-eval-prompt` sets eval rollouts per task.
- `--verifiers-v1-max-concurrent` limits concurrent episodes. Without it, Miles uses the smaller
  of the config's `max_concurrent` and aggregate SGLang server concurrency.
- `--verifiers-v1-num-eval-tasks` controls the number of eval tasks and defaults to
  `--rollout-batch-size`.
- `--eval-temperature`, `--eval-top-p`, `--eval-top-k`, `--eval-max-response-len`,
  `--eval-min-new-tokens`, `--eval-max-prompt-len`, and `--eval-max-context-len` provide Miles
  defaults and limits for eval rollouts.

Sampling follows Verifiers V1 override semantics: non-null values in the Verifiers sampling
config override the intercepted SDK request, and Miles rollout values provide the remaining
training or eval defaults. OpenAI token-limit and structured-output fields are translated to SGLang.
SGLang-native fields can be supplied in a request or in Verifiers sampling `extra_body` on
versions that expose it; a direct request field wins over its `extra_body` default.

## Supported Behavior

The adapter supports Chat Completions, Responses, and Anthropic V1 dialects, including streaming,
tool calls, token counting, user simulators, multi-turn episodes, environment runtimes, and every
graph branch. Each graph branch becomes a nested Miles training sample while preserving its V1
trace ID, rewards, metrics, stop condition, and errors.

Miles features remain available: deterministic inference seeds, oversampling and dynamic sampling
filters, sample and all-sample hooks, custom and group reward models, train/eval prompt and context
limits, image inputs, LoRA, named model routers, consistent-hash routing, prefill logprob
recomputation, routing and indexer replay, OPD student top-logprobs, speculative decoding metrics,
prefix-cache metrics, worker aborts, and weight-version tracking. Verifiers scoring is used unless a
Miles reward model is explicitly configured.

`--partial-rollout` is the one unsupported rollout mode. A V1 episode owns live harness and
environment state and has no resume contract for a partially executed episode; Miles' native
multi-turn rollout has the same restriction. The adapter rejects this combination during argument
validation instead of producing a trajectory that cannot be resumed correctly.
