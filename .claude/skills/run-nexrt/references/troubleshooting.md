# Troubleshooting

Run commands from the repository root with the Python selected by
`NEXRT_PYTHON`. Start with the [preflight checks](hardware-support.md) and the
first error in the server log.

| Symptom | What to check |
| --- | --- |
| Missing Python module | Confirm that `NEXRT_PYTHON` provides the dependencies listed in the README. |
| CUDA compilation failure | Inspect the compiler error and confirm CUDA Toolkit 12.9+ is selected. |
| Missing CP or MTP files | Check `NEXRT_LAYERS_DIR` and the extraction command in the README. MTP also requires `mtp.safetensors`. |
| Out of memory | Identify the failed allocation. Check weight sharing, available GPU memory, and the KV/request capacities passed to the server. |
| Slow startup or first request | Use the startup progress and worker logs to locate loading, compilation, or graph capture. A long delay alone does not establish that the server is healthy. |
| Nonzero device status or failed verification | Preserve the failing prompt and log, then check the weights, handoff state, and predictions as described below. |
| Handoff layout assertion | Check the SGLang version and KV/Mamba pool layouts expected by the adapter. |
| CUDA/NCCL error or worker exit | Inspect all worker logs for the first failure, then restart the worker group after resolving it. |
| `bad request` response | Correct the request fields or prompt length. The server rejects invalid requests before generation. |
| Port already in use | Check for an existing server; use `serve.sh --stop --port <N>` to stop the instance managed by the script. |

`--lens` creates length-controlled timing inputs and may repeat filler text.
Use `bridge_client.py --chat --text ...` or `--file ...` with meaningful prompts
when evaluating generated answers. File paths passed to the client are local
to the client.

## Unexpected output

Compare predictions over the same token history. Free-running continuations can
diverge after a single mismatch. The 1-batch smoke gate allows one mismatch in
eight predictions; it does not establish full numerical equivalence.

When the checkpoint or SGLang version changes, inspect FP8 scales and weight
layouts. The loader adapts `.weight_scale` names to `.weight_scale_inv`; the
reported number of wrapped loader classes does not prove that the scale values
loaded correctly. Check the registered scale parameters and checkpoint tensor
names when investigating a loading mismatch.

Weight sharing is only one part of correctness. Also examine the prefill state
and logits when a mismatch persists. Include the checkpoint, software versions,
request, and first failing log lines when reporting a problem.
