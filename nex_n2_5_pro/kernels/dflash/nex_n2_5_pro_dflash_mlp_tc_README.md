# DFlash MLP dependency ordering

## Bug: partial load moved before PDL synchronization

**Symptom:** Repeated complete DFlash rounds can produce different draft hidden
states and proposals. Final tokens may reconverge, so compare intermediate
rounds as well as the emitted sequence.

**Cause:** An ordinary `const`/`restrict` load of the down-GEMM partial in
`DfMlpDownReduceKernel` can be hoisted above `cudaGridDependencySynchronize`
by CUDA 13, allowing the consumer to read an unfinished producer buffer.
The independent residual read before the wait is intentional.

**Fix:** Load the partial with `ld.volatile.global.v2.f32` and a compiler memory
clobber after the dependency wait. The arithmetic, launch geometry and LL
protocol remain unchanged.

**Validation:** Repeat complete rounds using
[`test_decode_determinism.py`](../../tests/test_decode_determinism.py), and test
the live SGLang feature-prefill handoff. A fixed reduction order alone does not
ensure determinism when a producer/consumer read races.
