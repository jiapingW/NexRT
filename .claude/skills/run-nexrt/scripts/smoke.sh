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

# Send one request to a running NexRT service and check the answer end to end.
#
#   scripts/smoke.sh [--port 9411] [--host 127.0.0.1] [--text "..."] [--file PATH]
#                    [--max-new 64] [--seed N] [--raw] [--timeout 3600]
#                    [--log FILE] [--allow-unverified] [--python BIN]
#
# Passes only if the reply reports status=0 and the selected mode's correctness gate held.
# BS=1 permits one mismatch in eight teacher-forced tokens; MTP checks every sampled
# round exactly or permits one mismatch in the greedy comparison. DFlash checks target commits.
#
# The first request includes CUDA-graph capture in total latency, but graph capture completes
# before the selected mode's per-token timer begins.
set -uo pipefail

PORT=9411
HOST=127.0.0.1
TEXT='Explain what a CUDA graph is and why replaying one lowers per-token latency.'
FILE=""
MAX_NEW=64
SEED=""
CHAT=1
TIMEOUT=3600
LOG=""
ALLOW_UNVERIFIED=0
PYBIN="${NEXRT_PYTHON:-python3}"

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
        --port)              PORT="${2:?--port needs an argument}"; shift ;;
        --host)              HOST="${2:?--host needs an argument}"; shift ;;
        --text)              TEXT="${2:?--text needs an argument}"; shift ;;
        --file)              FILE="${2:?--file needs an argument}"; shift ;;
        --max-new)           MAX_NEW="${2:?--max-new needs an argument}"; shift ;;
        --seed)              SEED="${2:?--seed needs an argument}"; shift ;;
        --raw)               CHAT=0 ;;
        --timeout)           TIMEOUT="${2:?--timeout needs an argument}"; shift ;;
        --log)               LOG="${2:?--log needs an argument}"; shift ;;
        --allow-unverified)  ALLOW_UNVERIFIED=1 ;;
        --python)            PYBIN="${2:?--python needs an argument}"; shift ;;
        -h|--help)           usage ;;
        *) printf 'smoke: unknown argument %s (try --help)\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname -- "$SCRIPT_DIR")"
REPO_ROOT="$(cd -- "$SKILL_DIR/../../.." && pwd)"
RUN_DIR="${NEXRT_RUN_DIR:-$REPO_ROOT/.nexrt-run}"
CLIENT_PY="$REPO_ROOT/nex_n2_5_pro/sglang_prefill/bridge_client.py"
[ -n "$LOG" ] || LOG="$RUN_DIR/serve-$PORT.log"

# ------------------------------------------------------------------ wait for the listener
if [ -f "$LOG" ]; then
    printf 'smoke: waiting for "@@ LISTENING" in %s (up to %ss)\n' "$LOG" "$TIMEOUT"
    waited=0
    until grep -q '@@ LISTENING' "$LOG" 2>/dev/null; do
        if [ "$waited" -ge "$TIMEOUT" ]; then
            printf 'smoke: TIMEOUT -- the server never started listening. Last 20 log lines:\n\n' >&2
            tail -20 "$LOG" >&2
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    printf 'smoke: server is listening\n'
else
    printf 'smoke: no server log at %s; assuming a server is already up on %s:%s\n' "$LOG" "$HOST" "$PORT"
fi

# ------------------------------------------------------------------ one request
CMD=("$PYBIN" "$CLIENT_PY" --host "$HOST" --port "$PORT" --max-new "$MAX_NEW")
if [ -n "$FILE" ]; then CMD+=(--file "$FILE"); else CMD+=(--text "$TEXT"); fi
[ "$CHAT" -eq 1 ] && CMD+=(--chat)
[ -n "$SEED" ] && CMD+=(--seed "$SEED")

REPLY_FILE="$(mktemp "${TMPDIR:-/tmp}/nexrt-smoke.XXXXXX")"
trap 'rm -f "$REPLY_FILE"' EXIT

printf 'smoke: %s\n' "${CMD[*]}"
printf 'smoke: a cold first request performs checkpoint prefill and graph capture -- be patient\n\n'
if ! "${CMD[@]}" >"$REPLY_FILE" 2>&1; then
    printf 'smoke: FAIL -- the client could not complete the request\n\n' >&2
    cat "$REPLY_FILE" >&2
    exit 1
fi

# ------------------------------------------------------------------ verdict
ALLOW_UNVERIFIED="$ALLOW_UNVERIFIED" "$PYBIN" - "$REPLY_FILE" <<'PY'
import json
import os
import sys

allow_unverified = os.environ.get("ALLOW_UNVERIFIED") == "1"
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read().strip()
if not raw:
    print("smoke: FAIL -- the server sent an empty reply", file=sys.stderr)
    raise SystemExit(1)

try:
    reply = json.loads(raw.splitlines()[0])
except json.JSONDecodeError:
    print("smoke: FAIL -- the reply is not JSON:\n" + raw[:2000], file=sys.stderr)
    raise SystemExit(1)

if "error" in reply:
    print(f"smoke: FAIL -- the server rejected the request: {reply['error']}", file=sys.stderr)
    raise SystemExit(1)

problems = []
status = reply.get("status", -1)
if status != 0:
    problems.append(f"block status={status} (non-zero means a kernel reported an error; expected 0)")

mode = reply.get("mode", "mtp" if "mtp_us_per_tok" in reply else "bs1")
agree = reply.get("forced_agree", -1)
verified = True
correctness = ""
if mode == "dflash":
    checked = reply.get("verified_rounds")
    total = reply.get("rounds", 0)
    verified = checked is not None
    correctness = f"{checked}/{total} target commits" if verified else "not checked"
    if verified and (total < 1 or checked != total):
        problems.append(f"DFlash verification covered {checked}/{total} rounds")
elif mode == "mtp":
    value = str(agree)
    if value == "-":
        verified = False
        correctness = "not checked"
    else:
        try:
            label, counts = value.split(":", 1) if ":" in value else ("greedy", value)
            matched, total = (int(item) for item in counts.split("/", 1))
        except (TypeError, ValueError):
            problems.append(f"unrecognized MTP correctness result: {agree!r}")
            correctness = value
        else:
            correctness = value
            if label == "sample-exact" and matched != total:
                problems.append(f"sampled MTP verification covered {matched}/{total} rounds")
            elif label == "greedy" and (total < 1 or matched < total - 1):
                problems.append(f"MTP greedy agreement {matched}/{total} (at most one mismatch allowed)")
            elif label not in ("sample-exact", "greedy"):
                problems.append(f"unrecognized MTP correctness label: {label!r}")
else:
    try:
        agree = int(agree)
    except (TypeError, ValueError):
        problems.append(f"unrecognized BS=1 correctness result: {agree!r}")
        verified = False
        correctness = str(agree)
    else:
        verified = agree >= 0
        correctness = "not checked" if not verified else f"{agree}/8 teacher-forced"
        if verified and agree < 7:
            problems.append(f"teacher-forced agreement {agree}/8 vs SGLang's own greedy "
                            "continuation (>=7 required)")

if not verified:
    msg = "the server was started without --verify, so this request was NOT correctness-checked"
    if allow_unverified:
        print(f"smoke: WARN -- {msg}")
    else:
        problems.append(msg + "; restart serve.sh with --verify, or pass --allow-unverified")

n_gen = reply.get("generated_tokens", 0)
if n_gen < 2:
    problems.append(f"only {n_gen} token(s) generated -- the decode loop did not run")

us = reply.get("mtp_us_per_tok" if mode == "mtp" else "decode_us_per_tok") or 0.0
print()
print(f"  mode              {'DFlash M=8' if mode == 'dflash' else 'MTP M=4' if mode == 'mtp' else 'BS=1'}")
print(f"  prompt tokens     {reply.get('prompt_tokens')}")
print(f"  prefill           {reply.get('prefill_s')} s")
print(f"  handoff           {reply.get('gather_ms')} ms")
print(f"  decode            {us} us/tok" + (f"  ({1e6 / us:.1f} tok/s)" if us else ""))
print(f"  generated         {n_gen} tokens, stop={reply.get('stop')}")
print(f"  correctness       {correctness}")
print(f"  block status      {status}")
text = (reply.get("generated_text") or "").strip().replace("\n", " ")
print(f"  completion        {text[:300]}{'...' if len(text) > 300 else ''}")
print()
sys.stdout.flush()          # keep the summary above the verdict when both are piped

if problems:
    print("smoke: FAIL", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    print("\n  see ../references/troubleshooting.md", file=sys.stderr)
    raise SystemExit(1)

if not verified:
    print("smoke: PASS -- prefill, handoff and decode succeeded. Correctness was NOT checked.")
else:
    print("smoke: PASS -- prefill, handoff, decode and the correctness gate all succeeded.")
print("smoke: graph capture preceded the per-token timer; total first-request latency includes it.")
PY
