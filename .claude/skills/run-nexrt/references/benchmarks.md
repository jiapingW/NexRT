# Testing and measurements

Prepare the environment and auxiliary weights as described in the
[README](../../../../README.md), then run live prefill and decode in one
resident engine:

```bash
export NEXRT_SERVE_REPORT=/path/to/results.jsonl
"$NEXRT_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
    nex_n2_5_pro/sglang_prefill/bridge_serve.py \
    --lens 1024,8192,65536 --gen 64 --verify
```

Add `--mtp` or `--dflash` and configure its checkpoint as described in the
README. Each invocation uses one mode. `--lens` supplies length-controlled
timing inputs; use real prompts through the client or `--files` for answer
quality evaluation.

`--verify` checks 1-batch teacher-forced predictions against SGLang. Sampled MTP
and DFlash check committed blocks against CPU resolvers. These checks have
different scopes; see [Running NexRT](../SKILL.md) for their pass conditions.

For service measurements, omit `--verify` and use idle GPUs. The JSONL report
separates prefill, handoff, and decode timing. `ttft_s` currently measures the
server's preparation through handoff, before decode graph capture; the socket
returns the full response, so this field is not a client-observed streaming
TTFT. Decode timing also starts after capture. Measure the complete client
request when evaluating first-request latency.

The README's speculative throughput assumes the AL values shown. Actual
throughput depends on the draft checkpoint, prompt, and accepted-token count.
Service timing includes host-side generation work and differs from CUDA-graph
replay timing.
