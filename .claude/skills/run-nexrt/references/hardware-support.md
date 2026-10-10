# Hardware support

The supported configuration is **8×NVIDIA H100 or H200, TP=8**, with NVLS multicast support,
CUDA Toolkit 12.9+, and PyTorch 2.6+.

## Fixed execution geometry

`bridge.make_server_args` sets `tp_size=8`. Weight shards, the four-rank context-parallel
attention groups, and multicast reductions use this geometry.

`nex_n2_5_pro/kernel_build.py` compiles kernels for `sm_90a`. The full-attention implementations
use Hopper WGMMA. Changing a compiler flag alone does not establish support for another
architecture; that requires a separate kernel port and validation.

## Preflight checks

`scripts/preflight.sh` checks:

- Eight visible GPUs with compute capability 9.0, MIG disabled, and active NVLink links.
- Idle GPUs, unless `--allow-busy` was explicitly selected.
- CUDA Toolkit 12.9+, PyTorch 2.6+, and the symmetric-memory API used by NexRT.
- The selected Python environment's required imports, including SGLang for prefill.
- Configured weight directories, expected files, and writable build-lock storage.

These checks do not prove that NVLS multicast works, that a checkpoint's tensors match the
adapter, or that a new SGLang version preserves the expected layouts. Run the serving smoke
test to check the full pipeline. When a check fails, follow the
[troubleshooting guide](troubleshooting.md).
