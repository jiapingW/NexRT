# MTP

NexRT supports MTP decoding with M=4 target verification. Prepare the MTP
checkpoint with `tools/extract_layers.py --mtp-src` and select `--mtp` when
starting the example server. See the [README](../../README.md) for commands.

The MTP artifact contains `mtp.*` tensors and is stored as
`NEXRT_LAYERS_DIR/mtp.safetensors`. The target weights remain shared with
SGLang. During prefill, target hidden states initialize the draft context;
each captured round runs drafting, target verification, commit, and draft
extension.

Sampling defaults to temperature 0.6 and top-p 0.95. Sampled MTP accepts
per-request `temperature` and `top_p`; `--mtp-sampling greedy` selects greedy
verification. `--verify` checks sampled rounds against the independent CPU
resolver using the target distributions exported by the kernel. It does not
establish full-model equivalence with SGLang.

MTP requests do not accept `cache_key`. See the
[prefill bridge](../sglang_prefill/BRIDGE.md) for context and CP options.

## Source map

- [block.py](block.py): draft block and round execution.
- [serve.py](serve.py): capture, replay, stopping, and status checks.
- [weights.py](weights.py): MTP checkpoint preparation for each rank.
- [sampling_ref.py](sampling_ref.py): independent sampling reference.
- [sglang_mtp.py](../sglang_prefill/sglang_mtp.py): SGLang MTP-layer prefill and draft hand-off.
