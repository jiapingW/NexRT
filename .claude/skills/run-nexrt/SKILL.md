---
name: run-nexrt
description: Run and validate NexRT on supported GPU hosts using the example prefill/decode server; diagnose environment, startup, and request failures.
---

# Running NexRT

Use the [README](../../../README.md) for environment requirements, checkpoint
preparation, and the serve → smoke → stop commands. Run commands from the
repository root on the GPU host, using the Python selected by `NEXRT_PYTHON`.

The example server runs SGLang prefill and NexRT decode in eight TP workers.
Choose one decode mode at startup: 1-batch, MTP M=4, or DFlash M=8.

## Starting the server

`serve.sh` runs preflight before loading the model. To inspect the environment
without starting the server, run:

```bash
.claude/skills/run-nexrt/scripts/preflight.sh --python "$NEXRT_PYTHON"
```

`serve.sh --background` waits for the listener and reports startup progress.
Loading includes checkpoint reads, weight preparation, and compilation of the
kernels needed for initialization. The first request for a decode arm also
captures its CUDA graph; new prefill shapes may compile on first use.

The server defaults to port 9411 on `127.0.0.1`, with verification enabled by
`serve.sh`. To accept requests from other hosts on a trusted network, pass
`-- --host 0.0.0.0` to `serve.sh` and `--host GPU_HOST` to the client or smoke
script. Each script supports `--help`.

To send a prompt after startup:

```bash
"$NEXRT_PYTHON" nex_n2_5_pro/sglang_prefill/bridge_client.py \
    --port 9411 --chat --text "Explain CUDA graphs in one sentence." --max-new 128
```

## Checking the result

Run `smoke.sh` after startup. It requires clean device status and a passing
check for the selected mode:

- 1-batch: at least seven of eight teacher-forced predictions agree with SGLang.
- Sampled MTP: every completed round agrees with the CPU sampling resolver.
  Greedy MTP allows one teacher-forced token mismatch against the 1-batch path.
- DFlash: every committed block agrees with CPU greedy prefix resolution.

These checks cover different behavior. Passing them does not establish identical
free-running text across engines. Investigate failed checks using the same
prompt and token history; do not infer a rounding explanation from a mismatch
alone.

The reply includes generated text, device status, and stage timings. Graph
capture precedes the decode timer, so decode timing alone does not describe the
latency of the first request. See the [measurement guide](references/benchmarks.md)
for the meaning of the timing fields.

## Reference

- [Prefill bridge](../../../nex_n2_5_pro/sglang_prefill/BRIDGE.md): request options,
  weight sharing, context limits, and cache behavior.
- [Hardware support](references/hardware-support.md): preflight coverage.
- [Troubleshooting](references/troubleshooting.md): startup and request failures.
- [Testing and measurements](references/benchmarks.md): live validation and timing.
