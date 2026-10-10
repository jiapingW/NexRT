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

# Boot the NexRT example server: SGLang prefill -> device handoff -> NexRT decode,
# one process per GPU, listening on a TCP port for 1-batch decoding requests.
#
#   scripts/serve.sh [--port 9411] [--background] [--no-verify] [--gen-files 256]
#                    [--python BIN] [--log FILE] [--ready-timeout 3600]
#                    [--skip-preflight] [--allow-busy] [-- <extra bridge_serve.py args>]
#   scripts/serve.sh --stop [--port 9411]
#
# torchrun starts the eight TP workers directly and supervises their lifetime.
#
# Startup loads the checkpoint and builds NexRT from SGLang's live tensors.
# CUDA kernels compile on first use.
# The first request includes graph capture in total latency; decode_us_per_tok starts afterward.
set -uo pipefail

PORT=9411
BACKGROUND=0
VERIFY=1
GEN_FILES=256
PYBIN="${NEXRT_PYTHON:-python3}"
LOG=""
READY_TIMEOUT=3600
SKIP_PREFLIGHT=0
PREFLIGHT_ARGS=()
STOP=0
EXTRA=()

is_running() {
    local state
    state=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
    [[ -n "$state" && "$state" != *Z* ]]
}

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
        --port)           PORT="${2:?--port needs an argument}"; shift ;;
        --background|-b)  BACKGROUND=1 ;;
        --no-verify)      VERIFY=0 ;;
        --gen-files)      GEN_FILES="${2:?--gen-files needs an argument}"; shift ;;
        --python)         PYBIN="${2:?--python needs an argument}"; shift ;;
        --log)            LOG="${2:?--log needs an argument}"; shift ;;
        --ready-timeout)  READY_TIMEOUT="${2:?--ready-timeout needs an argument}"; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --allow-busy)     PREFLIGHT_ARGS+=(--allow-busy) ;;
        --stop)           STOP=1 ;;
        --)               shift; EXTRA=("$@"); break ;;
        -h|--help)        usage ;;
        *) printf 'serve: unknown argument %s (try --help)\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname -- "$SCRIPT_DIR")"
REPO_ROOT="$(cd -- "$SKILL_DIR/../../.." && pwd)"
RUN_DIR="${NEXRT_RUN_DIR:-$REPO_ROOT/.nexrt-run}"
PIDFILE="$RUN_DIR/serve-$PORT.pid"
SERVE_PY="$REPO_ROOT/nex_n2_5_pro/sglang_prefill/bridge_serve.py"
CLIENT_PY="$REPO_ROOT/nex_n2_5_pro/sglang_prefill/bridge_client.py"
mkdir -p "$RUN_DIR"
[ -n "$LOG" ] || LOG="$RUN_DIR/serve-$PORT.log"

# ------------------------------------------------------------------ --stop
if [ "$STOP" -eq 1 ]; then
    printf 'serve: asking the server on port %s to shut down\n' "$PORT"
    if "$PYBIN" "$CLIENT_PY" --port "$PORT" --shutdown 2>/dev/null; then
        printf 'serve: shutdown acknowledged\n'
        GRACE=60                    # it accepted; give the ranks time to unwind cleanly
    else
        printf 'serve: no server answered on port %s\n' "$PORT"
        GRACE=10                    # unreachable; do not wait out a full graceful shutdown
    fi
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE")
        for _ in $(seq 1 "$GRACE"); do
            is_running "$pid" || break
            sleep 1
        done
        if is_running "$pid"; then
            printf 'serve: pid %s still alive after %ss, sending SIGTERM\n' "$pid" "$GRACE"
            kill "$pid" 2>/dev/null
        fi
        rm -f "$PIDFILE"
    fi
    printf 'serve: check for orphaned rank processes with: nvidia-smi\n'
    exit 0
fi

# Apply the same mode requirements to preflight and the actual server.
for arg in "${EXTRA[@]}"; do
    case "$arg" in
        --no-cp|--mtp|--dflash) PREFLIGHT_ARGS+=("$arg") ;;
    esac
done

# ------------------------------------------------------------------ preflight
if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
    if ! "$SCRIPT_DIR/preflight.sh" --python "$PYBIN" "${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}"; then
        printf '\nserve: preflight failed -- refusing to boot. Fix the FAIL lines, or run with\n'
        printf 'serve: --skip-preflight only if you have deliberately accepted every one of them.\n' >&2
        exit 1
    fi
else
    printf 'serve: --skip-preflight given; hardware and environment gates were NOT checked\n'
fi

if [ -f "$PIDFILE" ] && is_running "$(cat "$PIDFILE")"; then
    printf 'serve: a server is already running on port %s (pid %s)\n' "$PORT" "$(cat "$PIDFILE")" >&2
    printf 'serve: stop it first with: %s --stop --port %s\n' "${BASH_SOURCE[0]}" "$PORT" >&2
    exit 1
fi

# ------------------------------------------------------------------ launch
CMD=("$PYBIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8
     "$SERVE_PY" --listen "$PORT" --gen-files "$GEN_FILES")
[ "$VERIFY" -eq 1 ] && CMD+=(--verify)
CMD+=("${EXTRA[@]+"${EXTRA[@]}"}")

printf '\nserve: %s\n' "${CMD[*]}"
printf 'serve: log -> %s\n' "$LOG"
if [ "$VERIFY" -eq 1 ]; then
    printf 'serve: --verify is on: every request runs the selected mode'"'"'s correctness gate.\n'
else
    printf 'serve: --verify is OFF: requests will not be correctness-checked.\n'
fi
cd "$REPO_ROOT" || exit 1

if [ "$BACKGROUND" -eq 0 ]; then
    printf 'serve: running in the foreground; Ctrl-C stops it. Use --background to script against it.\n\n'
    "${CMD[@]}" 2>&1 | tee "$LOG"
    exit "${PIPESTATUS[0]}"
fi

: >"$LOG"
nohup "${CMD[@]}" >>"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PIDFILE"
printf 'serve: started pid %s, waiting up to %ss for the listener\n\n' "$SERVER_PID" "$READY_TIMEOUT"

waited=0
while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    if grep -q '@@ LISTENING' "$LOG" 2>/dev/null; then
        printf 'serve: READY -- listening on port %s (pid %s)\n' "$PORT" "$SERVER_PID"
        grep -E '@@ BOOT|@@ LISTENING' "$LOG" | tail -8
        printf '\nserve: send a request:  %s/scripts/smoke.sh --port %s\n' "$SKILL_DIR" "$PORT"
        printf 'serve: stop the server: %s --stop --port %s\n' "${BASH_SOURCE[0]}" "$PORT"
        exit 0
    fi
    if ! is_running "$SERVER_PID"; then
        printf 'serve: FAILED -- the server exited before it started listening. Last 40 log lines:\n\n' >&2
        tail -40 "$LOG" >&2
        printf '\nserve: see ../references/troubleshooting.md\n' >&2
        rm -f "$PIDFILE"
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 60)) -eq 0 ]; then
        printf 'serve: still booting (%ss) -- %s\n' "$waited" "$(grep -E '@@ BOOT' "$LOG" | tail -1)"
    fi
done

printf 'serve: TIMEOUT after %ss without a listener; the server is still running as pid %s.\n' \
       "$READY_TIMEOUT" "$SERVER_PID" >&2
printf 'serve: inspect %s, then stop it with %s --stop --port %s\n' "$LOG" "${BASH_SOURCE[0]}" "$PORT" >&2
exit 1
