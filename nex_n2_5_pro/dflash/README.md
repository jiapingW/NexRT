# DFlash

NexRT supports DFlash decoding with M=8 target verification. Select `--dflash`
when starting the example server and set `NEXRT_DFLASH_DIR` to a matching BF16
draft checkpoint. See the [README](../../README.md) for launch commands.

The target model's MoE weights remain shared with SGLang. The M=8 verifier and
draft head reuse one packed target LM head. Target features collected during
prefill initialize the draft context; each decode round proposes tokens,
verifies them with the target, and commits the accepted prefix.

The server uses greedy target verification and the target tokenizer's stop
tokens. `--verify` checks each committed block against CPU greedy prefix
resolution. DFlash requests do not accept `cache_key` or per-request sampling
parameters.

CP applies to target attention. The draft retains its own attention layout;
`--no-cp` selects non-CP target attention. See the
[prefill bridge](../sglang_prefill/BRIDGE.md) for capacity options.

## Source map

- [config.py](config.py): supported draft geometry and checkpoint validation.
- [weights.py](weights.py): rank-local draft weights.
- [attention.py](attention.py), [block.py](block.py): draft attention and round execution.
- [serve.py](serve.py): graph replay, stopping, and status checks.
