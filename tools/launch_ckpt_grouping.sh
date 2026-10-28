#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same command manually on every selected node. NODE_IDS is ordered:
# the first node is rank 0 and the rendezvous host.
#
# Examples:
#   NODE_IDS=17 DRY_RUN=1 tools/launch_ckpt_grouping.sh examples/models/llama/run_llama.py
#   NODE_IDS=17,1 tools/launch_ckpt_grouping.sh examples/checkpoint_grouping/train_toy.py --max-steps 200
#   NODE_IDS=17,1,16,8 tools/launch_ckpt_grouping.sh examples/checkpoint_grouping/train_toy.py --max-steps 500

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ $# -lt 1 ]]; then
    printf 'Usage: NODE_IDS=17[,1,16,8] %s ENTRYPOINT [ARGS...]\n' "$0" >&2
    exit 2
fi

entrypoint="$1"
shift
if [[ ! -f "${entrypoint}" ]]; then
    printf 'ERROR: entrypoint does not exist under %s: %s\n' "${repo_root}" "${entrypoint}" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "${repo_root}/setup_experiment.sh"

node_ids_csv="${NODE_IDS:-17}"
IFS=',' read -r -a node_ids <<< "${node_ids_csv}"
nnodes="${NNODES:-${#node_ids[@]}}"
nproc_per_node="${NPROC_PER_NODE:-8}"
master_port="${MASTER_PORT:-29510}"

if [[ "${nnodes}" -ne "${#node_ids[@]}" ]]; then
    printf 'ERROR: NNODES=%s but NODE_IDS contains %s entries\n' "${nnodes}" "${#node_ids[@]}" >&2
    exit 2
fi

cluster_addr() {
    local node_id="$1"
    if [[ "${node_id}" =~ ^127\.0\.0\.([0-9]+)$ ]]; then
        node_id="${BASH_REMATCH[1]}"
    fi
    if [[ "${node_id}" =~ ^[0-9]+$ ]]; then
        printf '172.20.%s.2\n' "${node_id}"
    else
        printf '%s\n' "${node_id}"
    fi
}

if [[ -n "${MASTER_ADDR:-}" ]]; then
    master_addr="${MASTER_ADDR}"
elif [[ "${nnodes}" -eq 1 ]]; then
    master_addr=127.0.0.1
else
    master_addr="$(cluster_addr "${node_ids[0]}")"
fi

if [[ "${nnodes}" -gt 1 && "${master_addr}" == 127.* ]]; then
    printf 'ERROR: MASTER_ADDR=%s is loopback and cannot rendezvous across hosts.\n' "${master_addr}" >&2
    exit 2
fi

infer_node_rank() {
    local hostname_value local_addr idx expected_addr
    hostname_value="$(hostname)"
    for ((idx = 0; idx < ${#node_ids[@]}; idx++)); do
        expected_addr="$(cluster_addr "${node_ids[idx]}")"
        if [[ " $(hostname -I 2>/dev/null || true) " == *" ${expected_addr} "* ]]; then
            printf '%s\n' "${idx}"
            return 0
        fi
        if [[ "${expected_addr}" =~ ^172\.20\.([0-9]+)\.2$ ]] &&
           [[ "${hostname_value}" == *"172-20-${BASH_REMATCH[1]}-2"* ]]; then
            printf '%s\n' "${idx}"
            return 0
        fi
    done
    return 1
}

if [[ -n "${NODE_RANK:-}" ]]; then
    node_rank="${NODE_RANK}"
elif [[ "${nnodes}" -eq 1 ]]; then
    node_rank=0
else
    node_rank="$(infer_node_rank)" || {
        printf 'ERROR: cannot infer node rank from NODE_IDS=%s; set NODE_RANK explicitly.\n' "${node_ids_csv}" >&2
        exit 2
    }
fi

if (( node_rank < 0 || node_rank >= nnodes )); then
    printf 'ERROR: NODE_RANK=%s is outside [0, %s).\n' "${node_rank}" "${nnodes}" >&2
    exit 2
fi

export NODE_RANK="${node_rank}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED=1

# The repository's synthetic models do not need Hugging Face network access.
if [[ "${CKPT_OFFLINE:-1}" == 1 ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
log_dir="${LOG_DIR:-${repo_root}/runs/launch_logs/${run_id}}"
mkdir -p "${log_dir}"
log_file="${log_dir}/node-${node_rank}.log"

cmd=(
    torchrun
    "--nnodes=${nnodes}"
    "--nproc-per-node=${nproc_per_node}"
    "--node-rank=${node_rank}"
    "--master-addr=${master_addr}"
    "--master-port=${master_port}"
    "${entrypoint}"
    "$@"
)

printf 'node_rank=%s nnodes=%s gpus_per_node=%s master=%s:%s interface=%s\n' \
    "${node_rank}" "${nnodes}" "${nproc_per_node}" "${master_addr}" "${master_port}" "${NCCL_SOCKET_IFNAME}"
printf 'log=%s\ncommand:' "${log_file}"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

if [[ "${CKPT_LOG_TEE:-1}" == 1 ]]; then
    "${cmd[@]}" 2>&1 | tee "${log_file}"
else
    "${cmd[@]}" > "${log_file}" 2>&1
fi
