#!/usr/bin/env bash
set -Eeuo pipefail

# One-time environment creation for AdaCheck checkpoint-grouping research.
# Run this script once per node. For every experiment, source setup_experiment.sh.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_prefix="${CKPT_ENV_PREFIX:-/opt/conda/envs/sjw_ckpt_grouping}"
cache_root="${CKPT_CACHE_ROOT:-${script_dir}/runs/cache/sjw_ckpt_grouping}"
pypi_index="${PYPI_INDEX_URL:-https://pypi.mirrors.ustc.edu.cn/simple}"
pytorch_index="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
nnscaler_runtime_root="${script_dir}/third_party/nnscaler"
grouping_demo_root="${GROUPING_DEMO_ROOT:-}"
setup_marker="${env_prefix}/.ckpt_grouping_setup_complete"

find_conda_base() {
    if [[ -n "${CONDA_EXE:-}" ]]; then
        dirname "$(dirname "${CONDA_EXE}")"
    elif command -v conda >/dev/null 2>&1; then
        conda info --base
    elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
        printf '%s\n' /root/miniconda3
    elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
        printf '%s\n' /opt/conda
    else
        return 1
    fi
}

if [[ -f "${setup_marker}" ]]; then
    printf 'Environment is already complete: %s\n' "${env_prefix}"
    printf 'Use: source %q\n' "${script_dir}/setup_experiment.sh"
    exit 0
fi

if [[ -e "${env_prefix}" && "${CKPT_RESUME_SETUP:-0}" != 1 ]]; then
    printf 'ERROR: environment exists without a completion marker: %s\n' "${env_prefix}" >&2
    printf 'Inspect it, then rerun with CKPT_RESUME_SETUP=1 to resume installation.\n' >&2
    exit 3
fi

[[ -f "${script_dir}/setup.py" ]] || {
    printf 'ERROR: Merak setup.py not found under %s\n' "${script_dir}" >&2
    exit 2
}
[[ -f "${nnscaler_runtime_root}/PROVENANCE.md" ]] || {
    printf 'ERROR: vendored nnScaler provenance not found: %s\n' "${nnscaler_runtime_root}" >&2
    exit 2
}

conda_base="$(find_conda_base)" || {
    printf 'ERROR: conda was not found. Set CONDA_EXE or install Miniconda.\n' >&2
    exit 2
}
# shellcheck disable=SC1090
source "${conda_base}/etc/profile.d/conda.sh"

mkdir -p "${cache_root}/conda-pkgs" "${cache_root}/pip" \
    "${cache_root}/huggingface" "${cache_root}/torch"
export CONDA_PKGS_DIRS="${cache_root}/conda-pkgs"
export PIP_CACHE_DIR="${cache_root}/pip"
export PYTHONNOUSERSITE=1

if [[ ! -x "${env_prefix}/bin/python" ]]; then
    conda create -y -p "${env_prefix}" --override-channels \
        -c https://mirrors.ustc.edu.cn/anaconda/pkgs/main \
        python=3.10 pip
fi
conda activate "${env_prefix}"

python -m pip install --upgrade --index-url "${pypi_index}" \
    'pip<25' 'setuptools<76' wheel ninja

python -m pip install --index-url "${pytorch_index}" --extra-index-url "${pypi_index}" \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

python -m pip install --index-url "${pypi_index}" \
    numpy==1.26.4 \
    transformers==4.42.4 datasets==2.21.0 safetensors==0.4.5 \
    huggingface-hub python-dateutil psutil pynvml \
    tensorboard tensorboardX \
    scipy==1.13.1 networkx==3.3 pybind11==2.12.0 cppimport \
    dill importlib-resources more-itertools pulp pyyaml tqdm \
    matplotlib==3.9.2 pandas==2.2.2 rich \
    pyzmq requests flask flask-socketio \
    pytest pytest-xdist

# Keep repository code editable, but never install the broken modified nnScaler fork.
python -m pip install --no-deps --no-build-isolation -e "${script_dir}"

site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
nnscaler_alias="${site_packages}/nnscaler"
expected_target="$(readlink -f "${nnscaler_runtime_root}")"
if [[ -L "${nnscaler_alias}" ]]; then
    alias_target="$(readlink -f "${nnscaler_alias}")"
    if [[ "${alias_target}" != "${expected_target}" ]]; then
        if [[ "${alias_target}" == "$(readlink -f "${env_prefix}/src/nnscaler")" ]]; then
            rm "${nnscaler_alias}"
        else
            printf 'ERROR: %s already points to %s, expected %s\n' \
                "${nnscaler_alias}" "${alias_target}" "${expected_target}" >&2
            exit 4
        fi
    fi
elif [[ -e "${nnscaler_alias}" ]]; then
    printf 'ERROR: refusing to replace non-symlink nnScaler package: %s\n' \
        "${nnscaler_alias}" >&2
    exit 4
fi
if [[ ! -L "${nnscaler_alias}" ]]; then
    ln -s "${expected_target}" "${nnscaler_alias}"
fi

if [[ -n "${grouping_demo_root}" && -f "${grouping_demo_root}/setup.py" ]]; then
    python -m pip install --no-deps --no-build-isolation "${grouping_demo_root}"
else
    printf 'NOTE: optional grouping demo was not installed; set GROUPING_DEMO_ROOT explicitly.\n'
fi

NNSCALER_EXPECTED_ROOT="${nnscaler_runtime_root}" python -c 'from pathlib import Path; import os, sys, torch, transformers, datasets, numpy, nnscaler; resolved=Path(nnscaler.__file__).resolve(); expected=Path(os.environ["NNSCALER_EXPECTED_ROOT"]).resolve(); print(f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__} datasets={datasets.__version__} numpy={numpy.__version__}"); print(f"nnscaler={resolved}"); assert resolved == expected / "__init__.py"; assert torch.__version__.split("+")[0] == "2.3.1"; print(f"cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}")'
if [[ -n "${grouping_demo_root}" && -f "${grouping_demo_root}/setup.py" ]]; then
    python -c 'import bottleneck_match; print(f"bottleneck_match={bottleneck_match.__file__}")'
fi

python -m pip freeze > "${cache_root}/pip-freeze.txt"
printf 'completed_utc=%s\nmerak_root=%s\nnnscaler_vendored_root=%s\nnnscaler_source_digest=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${script_dir}" \
    "$(readlink -f "${nnscaler_runtime_root}")" \
    "a862749fd9c84625f9620eb5cf89abd445478308919a920396e332bb6dfa4b0d" \
    > "${setup_marker}"

printf '\nEnvironment ready: %s\n' "${env_prefix}"
printf 'For each experiment run:\n  source %q\n' "${script_dir}/setup_experiment.sh"
