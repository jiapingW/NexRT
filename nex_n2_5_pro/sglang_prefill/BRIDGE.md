# Prefill bridge

The example server runs SGLang prefill and NexRT decode in one process per GPU,
with TP=8. Each engine selects 1-batch, MTP M=4, or DFlash M=8 at startup.
See the [README](../../README.md) for requirements and launch commands.

## Weights and request state

SGLang loads the target checkpoint once. NexRT derives its decode tensors from
the live SGLang model. Routed MoE weights and their scales share storage;
DFlash's M=8 target kernels consume that layout directly. Projections that need
a different layout or normalization are prepared for the selected mode. The
embedding is gathered across vocabulary shards, and CP attention requires the
auxiliary files prepared by `tools/extract_layers.py`.

The adapter detaches tensors before deriving decode weights so those operations
do not retain autograd intermediates. SGLang's target weights remain resident
for subsequent prefill requests. MTP and DFlash load their additional weights
once during initialization.

During prefill, each completed chunk appends its KV rows to NexRT. Finalization
transfers the Gated DeltaNet state and initializes the decode slot. MTP and
DFlash also consume target features during prefill to initialize their draft
state. For MTP, SGLang runs its own MTP layer over every chunk (its EAGLE/NEXTN
draft extend); finalization copies the resulting draft KV rows to NexRT once. The selected decode graph is captured on first use and reused for
subsequent requests using that arm.

## Requests

The server accepts one JSON line per connection and returns one JSON response
with the complete generated text. Requests execute sequentially. The default
listener is `127.0.0.1:9411`; `--host` and `--listen` configure the address.

| Field | Meaning |
| --- | --- |
| `text` | Prompt text. |
| `chat` | Apply the target tokenizer's chat template. |
| `max_new` | Maximum output tokens, including the first token produced by prefill. Generation can stop earlier at a stop token or the context limit. |
| `seed` | Request sampling seed. |
| `cache_key` | Optional prompt-cache owner for 1-batch mode only. |
| `temperature`, `top_p` | Sampling overrides for 1-batch or sampled MTP. A zero temperature selects greedy decoding. |
| `top_k` | Candidate limit from 1 to 28, for 1-batch mode only. |

In 1-batch mode, omitted sampling parameters use the model's `generation_config.json`
(defaulting to temperature 0.6, top-p 0.95, and top-k 20 when absent). Parameters
apply from the first output token and can change between requests without
recapturing the decode graph.

The target tokenizer determines stop tokens. Invalid fields and oversized
prompts are rejected. `bridge_client.py --file` reads a file on the client and
sends its text; requests cannot select server-side files.
`{"cmd":"shutdown"}` stops the server.

The response includes generated token IDs, text, device status, and stage
timings. See [Testing and measurements](../../.claude/skills/run-nexrt/references/benchmarks.md)
for timing and verification scope.

For an OpenAI-compatible facade, start the bridge with `--listen 9411`, then run:

```bash
python3 -m nex_n2_5_pro.sglang_prefill.openai_serve \
  --backend-port 9411 --port 8000
```

It exposes `/v1/models`, `/v1/completions`, and `/v1/chat/completions` on
`127.0.0.1` by default. Streaming requests receive SSE deltas for committed
target, MTP, or DFlash tokens. `tools/stream_chat.py` provides an interactive
plain-text client. Add `--forward-sampling` to the facade to forward request
sampling parameters to a compatible backend; `top_k` is limited to 1-batch mode.
Pass `--cache-key NAME` only when using target decoding and
wanting prompt reuse.

## Mode options

- **1-batch:** the default mode, with automatic CP dispatch for long contexts.
- **MTP:** select `--mtp` and provide `mtp.safetensors` in `NEXRT_LAYERS_DIR`.
  SGLang loads it as a one-layer draft model whose KV pool shares the target's
  allocator.
  Sampling defaults to temperature 0.6 and top-p 0.95; `--mtp-sampling greedy`
  selects greedy verification. `--mtp-arm` and `--mtp-cp-threshold` control
  target attention dispatch.
- **DFlash:** select `--dflash` and set `NEXRT_DFLASH_DIR` to the draft checkpoint.
  SGLang also loads it as a draft model whose own code materializes the prompt
  context K/V from the captured target features. Target verification is greedy.
  Requests do not accept sampling overrides.

MTP and DFlash requests use fresh prompt state and do not accept `cache_key`.
Only one decode mode runs in an engine.

## Context and memory capacity

Pass server options after `serve.sh --`:

| Option | Purpose | Default |
| --- | --- | --- |
| `--max-prompt` | Maximum prompt length in tokens. | 262080 |
| `--max-total-tokens` | SGLang KV capacity. | 263168 |
| `--max-running-requests` | SGLang request/Mamba capacity. | 2 |
| `--mem-fraction-static` | SGLang static-memory budget fraction. | 0.72 |

KV capacity must exceed `--max-prompt`. Allocation also depends on the selected
mode and available device memory.

`--no-cp` removes the CP auxiliary-weight requirement. For 1-batch and MTP,
configure prompt and generation bounds below 49152 tokens; MTP also requires
`--mtp-arm noncp`. DFlash supports non-CP target attention at long context.
MTP still requires its own weight file when CP is disabled.

## Prompt-cache reuse

1-batch mode retains at most one prompt session. With a matching nonempty
`cache_key`, an identical prompt can reuse its prefill state; a prompt that
extends the exact retained token sequence can prefill only the suffix. Missing
keys, changed owners, and other token changes require full prefill. Shorter
prefix matches cannot reuse the Gated DeltaNet recurrent state.

Only the prompt boundary is retained, not the generated continuation. Verification
that advances SGLang invalidates the retained session. TP ranks agree on cache
transitions; divergent lifecycle state and CUDA/NCCL failures terminate the
worker group. Explicitly reset-safe request errors invalidate the cached state
before another request is accepted.

## Source map

- [bridge.py](bridge.py): prefill and state handoff.
- [ext_weights.py](ext_weights.py), [weight_loading.py](weight_loading.py):
  tensor adaptation and checkpoint loading.
- [bridge_serve.py](bridge_serve.py), [bridge_client.py](bridge_client.py):
  example server and client.
- [cache_lifecycle.py](cache_lifecycle.py), [cache_consensus.py](cache_consensus.py):
  prompt ownership and TP agreement.
