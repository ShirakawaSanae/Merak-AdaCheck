#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf 'usage: %s SESSION LOG_PATH -- COMMAND [ARG ...]\n' "$0" >&2
    exit 2
}

(( $# >= 4 )) || usage
session=$1
log_path=$2
shift 2
[[ $1 == -- ]] || usage
shift
(( $# > 0 )) || usage
[[ $session =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf 'ERROR: invalid tmux session name: %s\n' "$session" >&2
    exit 2
}
command -v tmux >/dev/null || {
    printf 'ERROR: tmux is unavailable\n' >&2
    exit 2
}
tmux has-session -t "$session" 2>/dev/null && {
    printf 'ERROR: tmux session already exists: %s\n' "$session" >&2
    exit 2
}

mkdir -p "$(dirname -- "$log_path")"
log_path=$(cd -- "$(dirname -- "$log_path")" && pwd)/$(basename -- "$log_path")
status_path="${log_path}.status"
[[ ! -e $log_path && ! -e $status_path ]] || {
    printf 'ERROR: log or status path already exists: %s\n' "$log_path" >&2
    exit 2
}

printf -v command_q '%q ' "$@"
printf -v shell_command \
    'set -o pipefail; %s > %q 2>&1; rc=$?; printf "exit_code=%%s\nfinished_utc=%%s\n" "$rc" "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)" > %q; exit "$rc"' \
    "$command_q" "$log_path" "$status_path"

tmux new-session -d -s "$session" "$shell_command"
printf 'session=%s\nlog=%s\nstatus=%s\n' "$session" "$log_path" "$status_path"
