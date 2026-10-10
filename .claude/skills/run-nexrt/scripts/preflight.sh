#!/usr/bin/env bash
# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NexRT preflight: refuse to start unless this machine can actually run this release.
#
#   scripts/preflight.sh [--allow-busy] [--no-sglang] [--python BIN] [--min-free-mib N] [--no-cp] [--mtp | --dflash]
#
# Exit 0  every hard gate passed -- safe to boot the service.
# Exit 1  at least one hard gate failed -- do NOT boot. Fix the reported cause.
#
# WARN never changes the exit code. FAIL always does.
#
# Kernels require eight Hopper GPUs (sm_90a, TP=8) with NVLS multicast.
set -uo pipefail

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then
    printf 'preflight: needs bash 4+ (found %s); run it as "bash %s"\n' \
           "${BASH_VERSION:-unknown}" "${BASH_SOURCE[0]}" >&2
    exit 2
fi

ALLOW_BUSY=0
REQUIRE_SGLANG=1
REQUIRE_CP=1
MODE=bs1
PYBIN="${NEXRT_PYTHON:-python3}"
MIN_FREE_MIB=1024          # per-GPU used-memory ceiling for "idle"
EXPECT_GPUS=8              # bridge_serve/make_server_args hard-code tp_size=8

usage() {
    awk '
        /^#   scripts\// {show = 1}
        show && /^#/ {sub(/^# ?/, ""); print; next}
        show {exit}
    ' "${BASH_SOURCE[0]}"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --allow-busy)    ALLOW_BUSY=1 ;;
        --no-cp)        REQUIRE_CP=0 ;;
        --mtp)          MODE=mtp ;;
        --dflash)       MODE=dflash ;;
        --no-sglang)     REQUIRE_SGLANG=0 ;;
        --python)        PYBIN="${2:?--python needs an argument}"; shift ;;
        --min-free-mib)  MIN_FREE_MIB="${2:?--min-free-mib needs an argument}"; shift ;;
        -h|--help)       usage ;;
        *) printf 'preflight: unknown argument %s (try --help)\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_BAD=$'\033[31m'; C_WARN=$'\033[33m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_OK=''; C_BAD=''; C_WARN=''; C_DIM=''; C_OFF=''
fi

FAILED=0
ok()   { printf '  %sPASS%s  %s\n' "$C_OK" "$C_OFF" "$*"; }
warn() { printf '  %sWARN%s  %s\n' "$C_WARN" "$C_OFF" "$*"; }
bad()  { printf '  %sFAIL%s  %s\n' "$C_BAD" "$C_OFF" "$*"; FAILED=1; }
info() { printf '        %s%s%s\n' "$C_DIM" "$*" "$C_OFF"; }
head2() { printf '\n%s\n' "$*"; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname -- "$SCRIPT_DIR")"
REPO_ROOT="$(cd -- "$SKILL_DIR/../../.." && pwd)"

printf 'NexRT preflight  (repo: %s)\n' "$REPO_ROOT"

# ---------------------------------------------------------------- repository
head2 'Repository'
if [ -f "$REPO_ROOT/nex_n2_5_pro/decode_engine.py" ] && [ -f "$REPO_ROOT/nex_n2_5_pro/sglang_prefill/bridge_serve.py" ]; then
    ok "NexRT source tree found"
else
    bad "not a NexRT checkout: $REPO_ROOT/nex_n2_5_pro/decode_engine.py is missing"
    info "run this script from inside a NexRT clone (.claude/skills/run-nexrt/scripts/preflight.sh)"
fi

n_cu=$(find "$REPO_ROOT/nex_n2_5_pro/kernels" -name '*.cu' 2>/dev/null | wc -l)
n_so=$(find "$REPO_ROOT/build/sm90a" -type f -name '*.so' 2>/dev/null | wc -l)
if [ "$n_so" -lt "$n_cu" ]; then
    info "kernels: $n_so/$n_cu cached -- the rest JIT-compile with nvcc on first boot (minutes, once)"
else
    info "kernels: $n_so/$n_cu cached (rebuilt automatically if a build input is newer)"
fi

if [ -w /tmp ]; then
    ok "/tmp writable (kernel_build.py takes its build locks there)"
else
    bad "/tmp is not writable -- kernel_build.py cannot take its nvcc build lock"
fi

# ---------------------------------------------------------------- GPUs
head2 'GPUs'
if ! command -v nvidia-smi >/dev/null 2>&1; then
    bad "nvidia-smi not found -- no NVIDIA driver visible in this environment"
    info "NexRT needs 8 physical H100 or H200 GPUs; a CPU-only box cannot run any part of the service"
else
    GPU_QUERY='index,name,compute_cap,memory.total,memory.used,mig.mode.current'
    mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits 2>/dev/null)
    N_GPU=${#GPU_ROWS[@]}

    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        warn "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES is set"
        info "the bridge uses tp_size=8 and indexes visible devices 0-7; ensure all eight GPUs are included"
    fi

    if [ "$N_GPU" -eq "$EXPECT_GPUS" ]; then
        ok "$N_GPU GPUs visible"
    else
        bad "$N_GPU GPUs visible, need exactly $EXPECT_GPUS"
        info "TP=8 is hard-coded (bridge.make_server_args tp_size=8, weights.py sharding, 4-rank CP groups)"
        info "this release cannot run on any other GPU count -- not with fewer, not with more"
    fi

    join() { local IFS=','; printf '%s' "$*"; }
    arch_idx=() arch_kinds=() mig_idx=() busy_idx=() busy_peak=0 busy_total=0
    for row in "${GPU_ROWS[@]+"${GPU_ROWS[@]}"}"; do
        IFS=',' read -r idx name cc mem_total mem_used mig_mode <<<"$row"
        idx=$(echo "$idx" | xargs); name=$(echo "$name" | xargs); cc=$(echo "$cc" | xargs)
        mem_total=$(echo "$mem_total" | xargs); mem_used=$(echo "$mem_used" | xargs)
        mig_mode=$(echo "$mig_mode" | xargs)

        case "$cc" in
            9.0) : ;;
            [0-9]*.[0-9]*)
                arch_idx+=("$idx"); arch_kinds+=("$cc ($name)") ;;
            *)
                warn "GPU $idx: driver did not report compute_cap ('$cc'); name is '$name'"
                case "$name" in
                    *H100*|*H200*) info "name looks like Hopper; treating as sm_90 but verify before trusting a run" ;;
                    *)      arch_idx+=("$idx"); arch_kinds+=("unknown ($name)") ;;
                esac
                ;;
        esac

        [ "$mig_mode" = "Enabled" ] && mig_idx+=("$idx")

        case "$mem_used" in
            ''|*[!0-9]*) : ;;
            *) if [ "$mem_used" -gt "$MIN_FREE_MIB" ]; then
                   busy_idx+=("$idx")
                   [ "$mem_used" -gt "$busy_peak" ] && busy_peak="$mem_used"
                   busy_total="$mem_total"
               fi ;;
        esac
    done

    busy=0
    if [ "$N_GPU" -gt 0 ]; then
        if [ "${#arch_idx[@]}" -eq 0 ]; then
            ok "all GPUs report compute capability 9.0 (Hopper / sm_90a)"
        else
            kinds=$(printf '%s\n' "${arch_kinds[@]+"${arch_kinds[@]}"}" | sort -u | paste -sd'; ' -)
            bad "GPU $(join "${arch_idx[@]}") report compute capability $kinds -- this release requires 9.0 (Hopper)"
        fi
        if [ "${#mig_idx[@]}" -eq 0 ]; then
            ok "MIG disabled on all GPUs"
        else
            bad "GPU $(join "${mig_idx[@]}") have MIG enabled -- disable MIG (TP=8 needs whole GPUs and NVLink)"
        fi
    else
        info "no GPU rows returned by nvidia-smi -- skipping the architecture and MIG checks"
    fi
    if [ "${#busy_idx[@]}" -gt 0 ]; then
        busy=1
        printf -v msg 'GPU %s not idle: up to %s MiB of %s MiB already in use' \
               "$(join "${busy_idx[@]}")" "$busy_peak" "$busy_total"
        if [ "$ALLOW_BUSY" -eq 1 ]; then warn "$msg"; else bad "$msg"; fi
    fi
    if [ "${#arch_idx[@]}" -ne 0 ]; then
        info "the shipped kernels are compiled -arch=sm_90a, and both full-attention kernels are"
        info "written in Hopper wgmma/TMA, which do not exist on Blackwell. There is no fallback"
        info "attention path in this repo. Read ../references/hardware-support.md."
        info "Do NOT edit the arch flags to make this pass -- the kernels will not run."
    fi

    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$apps" ]; then
        busy=1
        if [ "$ALLOW_BUSY" -eq 1 ]; then warn "compute processes are running on these GPUs:"
        else bad "compute processes are running on these GPUs (they must be idle):"; fi
        printf '%s\n' "$apps" | while IFS= read -r line; do info "$line"; done
        info "stop them, or re-run with --allow-busy if you accept sharing (timings become meaningless)"
    fi
    if [ "$busy" -eq 0 ]; then
        ok "all GPUs idle (no compute processes, <= ${MIN_FREE_MIB} MiB used each)"
    elif [ "$ALLOW_BUSY" -eq 1 ]; then
        warn "--allow-busy: proceeding on non-idle GPUs. Benchmark numbers from this machine are NOT valid."
    fi

    nvlink_out=$(nvidia-smi nvlink --status 2>/dev/null)
    n_links=$(printf '%s\n' "$nvlink_out" | grep -c 'GB/s')
    if [ "$n_links" -gt 0 ]; then
        ok "NVLink active ($n_links links reporting bandwidth)"
    else
        bad "no active NVLink links found -- the TP reduces need NVLink multicast (NVLS)"
        info "on an HGX/NVSwitch box also check: systemctl status nvidia-fabricmanager"
    fi
    if command -v systemctl >/dev/null 2>&1; then
        fm=$(systemctl is-active nvidia-fabricmanager 2>/dev/null)
        [ -n "$fm" ] && info "nvidia-fabricmanager: $fm"
    fi
fi

# ---------------------------------------------------------------- toolchain
head2 'Toolchain'
if ! command -v nvcc >/dev/null 2>&1; then
    bad "nvcc not found -- NexRT compiles its CUDA kernels at runtime, nvcc is mandatory"
    info "no .so files ship in the repo; every kernel is built on first boot"
else
    nvcc_ver=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
    if [ -z "$nvcc_ver" ]; then
        warn "could not parse the nvcc version from 'nvcc --version'"
    else
        nvcc_num=$(( 10#${nvcc_ver%%.*} * 100 + 10#${nvcc_ver##*.} ))
        if [ "$nvcc_num" -ge 1209 ]; then
            ok "CUDA Toolkit $nvcc_ver (>= 12.9)"
        else
            bad "CUDA Toolkit $nvcc_ver is below the required 12.9"
            info "older ptxas mis-targets '-arch=sm_90a' to sm_90 and rejects the wgmma kernels (see attn.py:114)"
        fi
    fi
fi

head2 'Python environment'
info "probing: $PYBIN"
probe=$("$PYBIN" - <<'PY' 2>/dev/null
def emit(k, v):
    print(f"{k}={v}")

try:
    import torch
    emit("torch", torch.__version__)
    emit("torch_cuda", torch.version.cuda or "none")
except Exception as exc:
    emit("torch_err", f"{type(exc).__name__}: {exc}")
for mod in ("safetensors", "sglang", "transformers"):
    try:
        m = __import__(mod)
        emit(mod, getattr(m, "__version__", "present"))
    except Exception:
        emit(mod, "")
try:
    import torch.distributed._symmetric_memory  # noqa: F401
    emit("symm", "yes")
except Exception:
    emit("symm", "no")
PY
)
getv() { printf '%s\n' "$probe" | sed -n "s/^$1=//p" | head -1; }

if [ -z "$probe" ]; then
    bad "'$PYBIN' could not be run, or it crashed on import"
    info "point at the right interpreter with --python BIN or NEXRT_PYTHON=/path/to/python"
else
    torch_ver=$(getv torch)
    if [ -z "$torch_ver" ]; then
        bad "torch not importable under $PYBIN: $(getv torch_err)"
    else
        t_maj=${torch_ver%%.*}; t_rest=${torch_ver#*.}; t_min=${t_rest%%.*}
        if [ "$(( 10#$t_maj * 100 + 10#$t_min ))" -ge 206 ]; then
            ok "torch $torch_ver (CUDA $(getv torch_cuda))"
        else
            bad "torch $torch_ver is below the required 2.6"
            info "NexRT needs torch 2.6+ for CUDA graphs and torch.distributed._symmetric_memory"
        fi
    fi

    [ -n "$(getv safetensors)" ] && ok "safetensors $(getv safetensors)" \
                                || bad "safetensors not importable -- weight loading needs it"

    if [ "$(getv symm)" = "yes" ]; then
        ok "torch.distributed._symmetric_memory importable (NVLS multicast reduces)"
    else
        bad "torch.distributed._symmetric_memory not importable -- the TP reduces cannot be built"
    fi

    sgl=$(getv sglang)
    if [ -n "$sgl" ]; then
        ok "sglang $sgl importable (the prefill engine)"
        [ -n "$(getv transformers)" ] && ok "transformers $(getv transformers)" \
                                     || warn "transformers not importable -- the bridge needs it for the tokenizer"
    elif [ "$REQUIRE_SGLANG" -eq 1 ]; then
        bad "sglang not importable under $PYBIN -- the bridge imports it at module load"
        info "run the bridge under the Python that provides SGLang"
        info "--no-sglang is only for low-level checks that do not start the service"
    else
        warn "sglang not importable (--no-sglang): the bridge service cannot run under this interpreter"
    fi
fi

# ---------------------------------------------------------------- weights
head2 'Weights and paths'
if [ -z "${NEXRT_MODEL_DIR:-}" ]; then
    bad "NEXRT_MODEL_DIR is not set"
    info "export NEXRT_MODEL_DIR=/path/to/nex_n2_5_pro"
elif [ ! -d "$NEXRT_MODEL_DIR" ]; then
    bad "NEXRT_MODEL_DIR=$NEXRT_MODEL_DIR is not a directory"
else
    miss=""
    for f in config.json model.safetensors.index.json; do
        [ -f "$NEXRT_MODEL_DIR/$f" ] || miss="$miss $f"
    done
    if [ -z "$miss" ]; then
        ok "NEXRT_MODEL_DIR=$NEXRT_MODEL_DIR"
    else
        bad "NEXRT_MODEL_DIR is missing:$miss"
        info "expected a standard HuggingFace sharded-safetensors checkpoint of nex_n2_5_pro"
    fi
fi

if [ "$REQUIRE_CP" -eq 1 ]; then
    missing_layers=0
    for L in $(seq 3 4 59); do
        if [ ! -f "${NEXRT_LAYERS_DIR:-}/layer.$L.safetensors" ]; then
            missing_layers=$((missing_layers + 1))
        fi
    done
    if [ "$missing_layers" -eq 0 ]; then
        ok "CP attention files found in $NEXRT_LAYERS_DIR"
    else
        bad "missing $missing_layers CP attention files; set NEXRT_LAYERS_DIR and run tools/extract_layers.py"
    fi
fi
if [ "$MODE" = mtp ] && [ ! -f "${NEXRT_LAYERS_DIR:-}/mtp.safetensors" ]; then
    bad "MTP requires NEXRT_LAYERS_DIR/mtp.safetensors; use tools/extract_layers.py --mtp-src"
fi
if [ "$MODE" = dflash ]; then
    for file in "${NEXRT_DFLASH_CONFIG:-${NEXRT_DFLASH_DIR:-}/config.json}" \
                "${NEXRT_DFLASH_WEIGHTS:-${NEXRT_DFLASH_DIR:-}/model.safetensors}"; do
        [ -f "$file" ] || bad "DFlash checkpoint file missing: $file; set NEXRT_DFLASH_DIR"
    done
fi

# ---------------------------------------------------------------- verdict
head2 'Verdict'
if [ "$FAILED" -eq 0 ]; then
    printf '  %sREADY%s  every hard gate passed.\n\n' "$C_OK" "$C_OFF"
    printf '  next: start the selected mode with serve.sh, then run smoke.sh.\n'
    printf '        Use the same Python, mode and CP options as this preflight.\n\n'
    exit 0
fi
printf '  %sBLOCKED%s  at least one hard gate failed -- do not boot the service.\n\n' "$C_BAD" "$C_OFF"
printf '  Fix the FAIL lines above. Do not work around the hardware gate:\n'
printf '  a run on the wrong GPU count or architecture does not produce a slower NexRT,\n'
printf '  it produces kernels that do not launch. See ../references/hardware-support.md.\n\n'
exit 1
