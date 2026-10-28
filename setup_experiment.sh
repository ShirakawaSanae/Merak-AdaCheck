#!/usr/bin/env bash

# Repeatable experiment settings. Edit defaults here, then source this file:
#   source setup_experiment.sh
#
# Command-line overrides still win, for example:
#   NODE_IDS=17,1 MASTER_PORT=29520 source setup_experiment.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'ERROR: this script must be sourced:\n  source %q\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

_ckpt_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CKPT_ENV_PREFIX="${CKPT_ENV_PREFIX:-/opt/conda/envs/sjw_ckpt_grouping}"
export CKPT_CACHE_ROOT="${CKPT_CACHE_ROOT:-${_ckpt_script_dir}/runs/cache/sjw_ckpt_grouping}"
export NNSCALER_ROOT="${NNSCALER_ROOT:-${_ckpt_script_dir}/third_party/nnscaler}"
export NNSCALER_PROVENANCE="${NNSCALER_ROOT}/PROVENANCE.md"
export GROUPING_DEMO_ROOT="${GROUPING_DEMO_ROOT:-}"
export CKPT_RUNS_ROOT="${CKPT_RUNS_ROOT:-${_ckpt_script_dir}/runs}"

# Cluster defaults; override these before sourcing or on the launcher command.
export NODE_IDS="${NODE_IDS:-17}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29510}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Local synthetic models are the default and need no Hugging Face network.
export CKPT_OFFLINE="${CKPT_OFFLINE:-1}"
if [[ "${CKPT_OFFLINE}" == 1 ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
else
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
fi

export PIP_CACHE_DIR="${CKPT_CACHE_ROOT}/pip"
export CONDA_PKGS_DIRS="${CKPT_CACHE_ROOT}/conda-pkgs"
export HF_HOME="${CKPT_CACHE_ROOT}/huggingface"
export TORCH_HOME="${CKPT_CACHE_ROOT}/torch"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Both import styles exist in the current artifact:
#   process_monitor.procguard.* and procguard.*
# nnscaler resolves to the provenance-recorded source vendored in this worktree.
export PYTHONPATH="${_ckpt_script_dir}/third_party:${_ckpt_script_dir}:${_ckpt_script_dir}/process_monitor"
export LD_LIBRARY_PATH="${CKPT_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset NNSCALER_PROJECT_ROOT

_ckpt_conda_base=""
if [[ -n "${CONDA_EXE:-}" ]]; then
    _ckpt_conda_base="$(dirname "$(dirname "${CONDA_EXE}")")"
elif command -v conda >/dev/null 2>&1; then
    _ckpt_conda_base="$(conda info --base)"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    _ckpt_conda_base=/root/miniconda3
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    _ckpt_conda_base=/opt/conda
elif [[ -x "${CKPT_ENV_PREFIX}/bin/python" && "${CKPT_RUNTIME_ONLY:-0}" == 1 ]]; then
    _ckpt_conda_base=""
else
    printf 'ERROR: conda was not found.\n' >&2
    return 2
fi

_ckpt_runtime_only="${CKPT_RUNTIME_ONLY:-0}"
if [[ ! -f "${CKPT_ENV_PREFIX}/.ckpt_grouping_setup_complete" &&
      "${_ckpt_runtime_only}" != 1 ]]; then
    printf 'ERROR: environment setup is incomplete: %s\n' "${CKPT_ENV_PREFIX}" >&2
    printf 'Run once: bash %q\n' "${_ckpt_script_dir}/setup_conda_env.sh" >&2
    printf 'For checkpoint-only experiments without nnScaler, set CKPT_RUNTIME_ONLY=1.\n' >&2
    return 3
fi
if [[ ! -x "${CKPT_ENV_PREFIX}/bin/python" ||
      ! -x "${CKPT_ENV_PREFIX}/bin/torchrun" ]]; then
    printf 'ERROR: checkpoint runtime binaries are missing under %s\n' \
        "${CKPT_ENV_PREFIX}" >&2
    return 3
fi

if [[ -n "${_ckpt_conda_base}" ]]; then
    # shellcheck disable=SC1090
    source "${_ckpt_conda_base}/etc/profile.d/conda.sh"
    conda activate "${CKPT_ENV_PREFIX}" || return 4
else
    export PATH="${CKPT_ENV_PREFIX}/bin:${PATH}"
fi

if [[ "${_ckpt_runtime_only}" == 1 ]]; then
    python -c 'import os, torch, Merak; assert torch.__version__.split("+")[0] == "2.3.1"; assert torch.cuda.is_available(); expected=int(os.environ.get("NPROC_PER_NODE", "1")); assert torch.cuda.device_count() >= expected, (torch.cuda.device_count(), expected); print(f"Checkpoint runtime validated: torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")' || return 5
fi

printf 'Experiment environment: %s\n' "${CKPT_ENV_PREFIX}"
printf 'Merak root: %s\n' "${_ckpt_script_dir}"
if [[ "${_ckpt_runtime_only}" == 1 ]]; then
    printf 'Mode: checkpoint-runtime-only (nnScaler not required)\n'
else
    printf 'nnScaler runtime: %s (vendored; provenance: %s)\n' \
        "${NNSCALER_ROOT}" "${NNSCALER_PROVENANCE}"
fi
printf 'Nodes: %s; GPUs/node: %s; NCCL interface: %s\n' \
    "${NODE_IDS}" "${NPROC_PER_NODE}" "${NCCL_SOCKET_IFNAME}"

unset _ckpt_conda_base _ckpt_runtime_only _ckpt_script_dir
