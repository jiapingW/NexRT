# Decode determinism

NexRT provides deterministic decode by default for BS=1, MTP M=4, and
DFlash M=8. No extra configuration is required.

Repeated decode produces the same tokens when these remain unchanged:

- Decode mode, CP configuration, hardware, and software build.
- Target and draft weights.
- Complete prefill state, including target KV/GDN and any draft state.
- Sampling parameters and request seed.

MTP and DFlash also reproduce the number of tokens accepted in each round.

The guarantee begins at the prefill handoff. SGLang controls prefill, including
draft prompt processing. Reusing a seed alone is insufficient if the prefill
state changes. Outputs may differ across configurations or from SGLang decode.

## Request seeds

For sampled requests, pass `--seed N` to [bridge_client.py](sglang_prefill/bridge_client.py).
Without an explicit seed, the example server assigns one per request.
DFlash serving currently uses greedy verification.

## Validation

The [CP reduction test](tests/test_cp_reduce_determinism.py) checks fixed-order
TP8 reduction without model weights. The
[full-model test](tests/test_decode_determinism.py) checks repeated decode from
prepared saved state, including token decisions and speculative acceptance.
