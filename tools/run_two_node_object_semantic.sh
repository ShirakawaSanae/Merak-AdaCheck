#!/usr/bin/env bash
set -Eeuo pipefail

# One arm of a fixed node17,node8 SemanticFixed recovery gate.
# This driver must itself be owned by the caller's tmux session.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

arm="${1:?usage: $0 nockpt|adacheck_fixed|host_global_fixed|m1 [output-root]}"
case "${arm}" in nockpt|adacheck_fixed|host_global_fixed|m1) ;; *) exit 2 ;; esac
output_root="${2:-${repo_root}/runs/two-node-v2/object-semantic-${arm}-$(date -u +%Y%m%dT%H%M%SZ)}"

# This is intentionally not a general two-host launcher.
local_node=17
remote_node=8
remote_host=172.20.8.2
model="${CKPT_OBJECT_MODEL:-gpt2}"
detector_config="${CKPT_OBJECT_DETECTOR_CONFIG:-${repo_root}/artifacts/ae1028/runs/two-node-v2/detector-gpt2-bf16-pp4tp1dp4-world-object-r115-20260913/experiment-config.json}"
pp="${CKPT_OBJECT_PP:-4}"
tp="${CKPT_OBJECT_TP:-1}"
dp="${CKPT_OBJECT_DP:-4}"
mesh_order="${CKPT_OBJECT_MESH_ORDER:-pp-dp}"
sequence_length="${CKPT_OBJECT_SEQUENCE_LENGTH:-128}"
gradient_accumulation_steps="${CKPT_OBJECT_GRADIENT_ACCUMULATION_STEPS:-1}"
source_steps="${CKPT_OBJECT_SOURCE_STEPS:-10}"
continue_steps="${CKPT_OBJECT_CONTINUE_STEPS:-5}"
reference_steps=$((source_steps + continue_steps))
master_port="${MASTER_PORT:-29940}"
service_port="${SERVICE_PORT:-29950}"
service_timeout_seconds="${CKPT_OBJECT_SERVICE_TIMEOUT_SECONDS:-600}"
service_max_mib="${SERVICE_MAX_MIB:-163840}"
# Conservative full-state catalog upper bound, not expected active-gradient traffic.
estimated_service_bytes="${CKPT_OBJECT_ESTIMATED_SERVICE_BYTES:-114920411984}"
minimum_service_mib="${CKPT_OBJECT_MINIMUM_SERVICE_MIB:-131072}"
if [[ -v CKPT_OBJECT_ESTIMATED_SERVICE_BYTES || -v CKPT_OBJECT_MINIMUM_SERVICE_MIB ]]; then
    [[ -v CKPT_OBJECT_ESTIMATED_SERVICE_BYTES && -v CKPT_OBJECT_MINIMUM_SERVICE_MIB ]] || {
        printf 'ERROR: provide both CKPT_OBJECT_ESTIMATED_SERVICE_BYTES and CKPT_OBJECT_MINIMUM_SERVICE_MIB.\n' >&2
        exit 2
    }
    capacity_estimate="explicit operator-provided retention estimate; not measured active-gradient traffic or a performance setting"
else
    [[ "${pp}/${tp}/${dp}/${mesh_order}" == "4/1/4/pp-dp" ]] || {
        printf 'ERROR: non-default layouts require an explicit service retention estimate and minimum.\n' >&2
        exit 2
    }
    capacity_estimate="conservative r115 full-state catalog upper bound; not measured active-gradient traffic or a performance setting"
fi
max_inflight="${CKPT_OBJECT_MAX_INFLIGHT:-${source_steps}}"
base_upload_parallelism="${CKPT_OBJECT_BASE_UPLOAD_PARALLELISM:-1}"
base_mode="${CKPT_OBJECT_BASE_MODE:-dp_deduplicated_v1}"
restore_enabled="${RUN_RESTORE:-1}"
failed_service_node="${FAILED_SERVICE_NODE_ID:-8}"
learning_rate=5e-5
weight_decay=0.0
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-8
restore_factory="${CKPT_OBJECT_RECOVERY_FACTORY:-Merak.core.checkpoint.object_semantic_restore:build_object_semantic_restore_controller}"
python="${PYTHON:-/opt/conda/envs/sjw_ckpt_grouping/bin/python}"
ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)
local_service_pid=""
remote_service_pid=""
remote_service_wait_pid=""
active_local_group=""
active_remote_group=""
active_local_pid=""
active_remote_ssh_pid=""

[[ "${CKPT_LOCAL_NODE_ID:-17}" == 17 && "${CKPT_REMOTE_NODE_ID:-8}" == 8 ]] || {
    printf 'ERROR: this gate is fixed to ordered nodes 17,8.\n' >&2; exit 2; }
[[ "${model}" == gpt2 || "${model}" == gpt_6b || "${model}" == llama_3_8b ]] || exit 2
[[ "${master_port}" != "${service_port}" &&
   "${pp}" =~ ^[1-9][0-9]*$ && "${tp}" =~ ^[1-9][0-9]*$ && "${dp}" =~ ^[1-9][0-9]*$ &&
   "${sequence_length}" =~ ^[1-9][0-9]*$ && "${gradient_accumulation_steps}" =~ ^[1-9][0-9]*$ &&
   "${source_steps}" =~ ^[1-9][0-9]*$ && "${continue_steps}" =~ ^[1-9][0-9]*$ &&
   "${max_inflight}" =~ ^[1-9][0-9]*$ && "${base_upload_parallelism}" =~ ^[1-9][0-9]*$ &&
   "${service_max_mib}" =~ ^[1-9][0-9]*$ &&
   "${service_timeout_seconds}" =~ ^[1-9][0-9]*$ &&
   "${minimum_service_mib}" =~ ^[1-9][0-9]*$ && "${estimated_service_bytes}" =~ ^[1-9][0-9]*$ &&
   "${mesh_order}" == pp-dp &&
   ( "${base_mode}" == rank_local_v1 || "${base_mode}" == dp_deduplicated_v1 ) ]] || exit 2
(( pp * tp * dp == 16 && (pp == 2 || pp == 4) && (tp == 1 || tp == 2) )) || exit 2
if [[ "${arm}" == nockpt ]]; then restore_enabled=0; fi
(( service_max_mib >= minimum_service_mib )) || {
    printf 'ERROR: SERVICE_MAX_MIB must be at least %s for the configured retention estimate.\n' "${minimum_service_mib}" >&2; exit 2; }
(( max_inflight >= source_steps )) || {
    printf 'ERROR: CKPT_OBJECT_MAX_INFLIGHT must cover all %s source steps for this final-drain gate.\n' "${source_steps}" >&2; exit 2; }
[[ "${restore_enabled}" == 0 || "${restore_enabled}" == 1 ]] || exit 2
[[ "${failed_service_node}" == 17 || "${failed_service_node}" == 8 ]] || exit 2
[[ -f "${detector_config}" && ! -e "${output_root}" ]] || {
    printf 'ERROR: detector config missing or output already exists.\n' >&2; exit 2; }

validate_config() {
    "${python}" - "${detector_config}" "${model}" "${pp}" "${tp}" "${dp}" "${mesh_order}" <<'PY'
import json, sys
from pathlib import Path
path, model, pp, tp, dp, mesh = sys.argv[1:]
v = json.loads(Path(path).read_text())
p = v.get("parallelism")
topology = v.get("workload", {}).get("rank_topology")
if (
    v.get("model_argument") != model
    or v.get("precision") != "bf16"
    or v.get("optimizer") != {
        "name": "torch.optim.AdamW", "foreach": False, "fused": False
    }
    or v.get("detector_comparison_scope") != "world"
    or not isinstance(p, dict)
    or (p.get("world_size"), p.get("pipeline"), p.get("tensor"), p.get("data"), p.get("mesh_order"))
       != (16, int(pp), int(tp), int(dp), mesh)
    or not isinstance(topology, list)
    or sorted(item.get("rank") for item in topology if isinstance(item, dict)) != list(range(16))
):
    raise ValueError("detector config does not match the selected 16-rank topology")
PY
}
validate_config

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY_RUN arm=%s nodes=17,8 model=%s layout=PP%s/TP%s/DP%s/%s seq=%s ga=%s base_mode=%s base_upload_parallelism=%s detector=%s restore=%s\n' \
        "${arm}" "${model}" "${pp}" "${tp}" "${dp}" "${mesh_order}" \
        "${sequence_length}" "${gradient_accumulation_steps}" "${base_mode}" "${base_upload_parallelism}" "${detector_config}" "${restore_enabled}"
    exit 0
fi
[[ -n "${TMUX:-}" ]] || { printf 'ERROR: run this driver inside the caller tmux session.\n' >&2; exit 2; }
[[ " $(hostname -I 2>/dev/null || true) " == *" 172.20.17.2 "* ]] || {
    printf 'ERROR: run this driver on node17.\n' >&2; exit 2; }

mkdir -p "${output_root}"/{logs,services,source/{base,updates,metrics,digests,profiles,recovery_metrics},restore/digests}
cp -- "${detector_config}" "${output_root}/detector-experiment-config.json"
git rev-parse HEAD > "${output_root}/git-commit.txt"
git status --porcelain > "${output_root}/git-status.txt"
git diff --binary | sha256sum | awk '{print $1}' > "${output_root}/tracked-diff-sha256.txt"
{
    date -u +utc=%Y-%m-%dT%H:%M:%SZ; hostname; uname -a; lscpu
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} > "${output_root}/system-node17.txt"
ssh "${ssh_options[@]}" "${remote_host}" \
    'date -u +utc=%Y-%m-%dT%H:%M:%SZ; hostname; uname -a; lscpu; nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader' \
    > "${output_root}/system-node8.txt"

"${python}" - "${output_root}" "${arm}" "${restore_enabled}" "${service_port}" "${master_port}" "${max_inflight}" "${base_upload_parallelism}" "${base_mode}" "${service_max_mib}" "${estimated_service_bytes}" "${minimum_service_mib}" "${capacity_estimate}" "${failed_service_node}" "${model}" "${pp}" "${tp}" "${dp}" "${mesh_order}" "${sequence_length}" "${gradient_accumulation_steps}" "${source_steps}" "${continue_steps}" "${reference_steps}" "${learning_rate}" "${weight_decay}" "${adam_beta1}" "${adam_beta2}" "${adam_epsilon}" "${service_timeout_seconds}" <<'PY'
import json, sys
from pathlib import Path
root, arm, restore, service, master, max_inflight, base_upload_parallelism, base_mode, capacity, estimate, minimum, estimate_note, failed, model, pp, tp, dp, mesh, sequence_length, gradient_accumulation_steps, source_steps, continue_steps, reference_steps, lr, decay, beta1, beta2, epsilon, service_timeout = sys.argv[1:]
checkpoint_enabled = arm != "nockpt"
value = {"schema_version": 1, "arm": arm, "run_id": Path(root).name,
         "nodes": [17, 8], "world_size": 16, "ranks_per_host": 8,
         "model": model, "parallelism": {"pp": int(pp), "tp": int(tp), "dp": int(dp), "mesh_order": mesh},
         "workload": {"sequence_length": int(sequence_length), "per_device_train_batch_size": 1,
             "gradient_accumulation_steps": int(gradient_accumulation_steps), "seed": 20260913, "precision": "bf16"},
         "checkpoint_enabled": checkpoint_enabled,
         "source_steps": (int(source_steps) if checkpoint_enabled else int(reference_steps)),
         "reference_steps": int(reference_steps),
         "restore_enabled": restore == "1", "restore_epoch": (int(source_steps) if checkpoint_enabled else None),
         "restore_continue_steps": (int(continue_steps) if checkpoint_enabled else 0),
         "failed_service_node": (int(failed) if checkpoint_enabled else None),
         "surviving_host": ((1 if int(failed) == 17 else 0) if checkpoint_enabled else None),
         "scheduler": "constant", "optimizer": {"name": "scalar_adamw", "learning_rate": float(lr),
             "weight_decay": float(decay), "beta1": float(beta1), "beta2": float(beta2),
             "epsilon": float(epsilon)},
         "service_port": int(service), "master_port": int(master),
         "service_timeout_seconds": int(service_timeout),
         "base_mode": base_mode,
         "base_upload_parallelism": int(base_upload_parallelism),
         "max_inflight": (int(max_inflight) if checkpoint_enabled else None),
         "cadence_claim": ("final-drain allowed; strict_1c1s_valid reported separately" if checkpoint_enabled else "not applicable"),
         "recovery_claim": ("fresh restore completion; suite-level trajectory oracle required" if checkpoint_enabled else "uninterrupted reference"),
         "service_capacity": {"configured_mib": int(capacity), "minimum_mib": int(minimum),
                              "estimated_retained_bytes": int(estimate),
                              "estimate": estimate_note}}
Path(root, "run-contract.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
printf '%q ' env RUN_RESTORE="${restore_enabled}" FAILED_SERVICE_NODE_ID="${failed_service_node}" \
    CKPT_OBJECT_MODEL="${model}" CKPT_OBJECT_PP="${pp}" CKPT_OBJECT_TP="${tp}" CKPT_OBJECT_DP="${dp}" CKPT_OBJECT_MESH_ORDER="${mesh_order}" \
    CKPT_OBJECT_SEQUENCE_LENGTH="${sequence_length}" CKPT_OBJECT_GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}" \
    CKPT_OBJECT_SOURCE_STEPS="${source_steps}" CKPT_OBJECT_CONTINUE_STEPS="${continue_steps}" CKPT_OBJECT_MAX_INFLIGHT="${max_inflight}" \
    CKPT_OBJECT_BASE_UPLOAD_PARALLELISM="${base_upload_parallelism}" \
    CKPT_OBJECT_BASE_MODE="${base_mode}" \
    CKPT_OBJECT_DETECTOR_CONFIG="${detector_config}" \
    CKPT_OBJECT_SERVICE_TIMEOUT_SECONDS="${service_timeout_seconds}" \
    CKPT_OBJECT_ESTIMATED_SERVICE_BYTES="${estimated_service_bytes}" CKPT_OBJECT_MINIMUM_SERVICE_MIB="${minimum_service_mib}" \
    MASTER_PORT="${master_port}" SERVICE_PORT="${service_port}" bash "$0" "${arm}" "${output_root}" \
    > "${output_root}/command.txt"
printf '\n' >> "${output_root}/command.txt"

preflight() {
    local host="$1" destination="$2" gpu jobs ports port_query_ok=1
    gpu="$(ssh "${ssh_options[@]}" "${host}" 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true')"
    jobs="$(ssh "${ssh_options[@]}" "${host}" "pgrep -af '[t]orchrun|[r]un_workload.py|[r]un_object_replica_service.py' || true")"
    ports="$(ssh "${ssh_options[@]}" "${host}" "if command -v ss >/dev/null 2>&1; then listeners=\$(ss -ltn) || exit; elif command -v netstat >/dev/null 2>&1; then listeners=\$(netstat -ltn) || exit; else exit 127; fi; printf '%s\\n' \"\${listeners}\" | grep -E '[:.](${service_port}|${master_port})[[:space:]]' || true")" || port_query_ok=0
    { printf 'gpu_compute_apps=%q\nlauncher_or_service_processes=%q\nexperiment_port_listeners=%q\nport_query_ok=%s\n' "${gpu}" "${jobs}" "${ports}" "${port_query_ok}"; } > "${destination}"
    [[ "${port_query_ok}" == 1 && -z "${gpu//[[:space:]]/}" && -z "${jobs//[[:space:]]/}" && -z "${ports//[[:space:]]/}" ]]
}
if ! preflight 172.20.17.2 "${output_root}/preflight-node17.txt" ||
   ! preflight "${remote_host}" "${output_root}/preflight-node8.txt"; then
    printf 'ERROR: GPU, experiment process, or experiment port is busy; see preflight files.\n' >&2
    exit 1
fi
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
    printf 'preflight_complete=1\n' > "${output_root}/PREFLIGHT_COMPLETE"
    exit 0
fi

cleanup() {
    [[ "${active_local_pid}" =~ ^[0-9]+$ ]] && kill -TERM "${active_local_pid}" 2>/dev/null || true
    [[ "${active_remote_ssh_pid}" =~ ^[0-9]+$ ]] && kill -TERM "${active_remote_ssh_pid}" 2>/dev/null || true
    [[ "${active_local_group}" =~ ^[0-9]+$ ]] && kill -TERM -- "-${active_local_group}" 2>/dev/null || true
    [[ "${active_remote_group}" =~ ^[0-9]+$ ]] && ssh "${ssh_options[@]}" "${remote_host}" "kill -TERM -- -${active_remote_group} 2>/dev/null || true" || true
    [[ "${local_service_pid}" =~ ^[0-9]+$ ]] && kill -TERM "${local_service_pid}" 2>/dev/null || true
    [[ "${remote_service_pid}" =~ ^[0-9]+$ ]] && ssh "${ssh_options[@]}" "${remote_host}" "kill -TERM ${remote_service_pid} 2>/dev/null || true" || true
    [[ "${active_local_pid}" =~ ^[0-9]+$ ]] && wait "${active_local_pid}" 2>/dev/null || true
    [[ "${active_remote_ssh_pid}" =~ ^[0-9]+$ ]] && wait "${active_remote_ssh_pid}" 2>/dev/null || true
    [[ "${remote_service_wait_pid}" =~ ^[0-9]+$ ]] && wait "${remote_service_wait_pid}" 2>/dev/null || true
}
on_exit() {
    local exit_code=$?
    cleanup
    if (( exit_code != 0 )); then
        printf 'failed_utc=%s exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${exit_code}" > "${output_root}/FAILED"
    fi
}
trap on_exit EXIT
trap 'exit 130' HUP INT TERM

start_service() {
    local node="$1" host="$2" ready="$3" pid_file="$4" log="$5"
    if [[ "${node}" == 17 ]]; then
        env PYTHONPATH=.:process_monitor "${python}" examples/checkpoint_grouping/run_object_replica_service.py \
            --bind 0.0.0.0 --port "${service_port}" --max-store-mib "${service_max_mib}" \
            --retained-epochs 16 --torch-num-threads 0 --ready-file "${ready}" > "${log}" 2>&1 &
        local_service_pid=$!; printf '%s\n' "${local_service_pid}" > "${pid_file}"
    else
        ssh "${ssh_options[@]}" "${host}" "cd $(printf '%q' "${repo_root}") && echo \$\$ > $(printf '%q' "${pid_file}") && exec env PYTHONPATH=.:process_monitor $(printf '%q' "${python}") examples/checkpoint_grouping/run_object_replica_service.py --bind 0.0.0.0 --port $(printf '%q' "${service_port}") --max-store-mib $(printf '%q' "${service_max_mib}") --retained-epochs 16 --torch-num-threads 0 --ready-file $(printf '%q' "${ready}")" > "${log}" 2>&1 &
        remote_service_wait_pid=$!
    fi
}
if [[ "${arm}" != nockpt ]]; then
    start_service 17 172.20.17.2 "${output_root}/services/node17.ready" "${output_root}/services/node17.pid" "${output_root}/services/node17.log"
    start_service 8 "${remote_host}" "${output_root}/services/node8.ready" "${output_root}/services/node8.pid" "${output_root}/services/node8.log"
    for _ in {1..120}; do
        [[ -s "${output_root}/services/node17.ready" && -s "${output_root}/services/node8.ready" ]] && break
        sleep 0.5
    done
    [[ -s "${output_root}/services/node17.ready" && -s "${output_root}/services/node8.ready" ]] || exit 1
    remote_service_pid="$(<"${output_root}/services/node8.pid")"
fi

monotonic_ns() {
    "${python}" -c 'import time; print(time.monotonic_ns())'
}

service_stats() {
    local destination="$1"
    "${python}" - "${destination}" "${service_port}" <<'PY'
import json, sys
from Merak.core.checkpoint.replica_service import ReplicaServiceClient
out, port = sys.argv[1], int(sys.argv[2])
value = {}
for name, endpoint in (("node17", f"172.20.17.2:{port}"), ("node8", f"172.20.8.2:{port}")):
    try:
        stats, metrics = ReplicaServiceClient.from_endpoint(endpoint, timeout_seconds=10).stats_with_metrics()
        value[name] = {"endpoint": endpoint, "status": "ok", "stats": stats,
                       "request_bytes": metrics.request_bytes, "response_bytes": metrics.response_bytes,
                       "server_ms": metrics.server_ms}
    except Exception as error:
        value[name] = {"endpoint": endpoint, "status": "unavailable",
                       "error": f"{type(error).__name__}: {error}"}
open(out, "w").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
}

run_phase() {
    local phase="$1" max_steps="$2" mode="$3" restore_epoch="${4:-}"
    local root="${output_root}/${phase}"
    local -a envs=(
        "CKPT_EXPERIMENT_ARM=${arm}" "CKPT_RANKS_PER_NODE=8"
        "CKPT_FINAL_DIGEST_DIR=${root}/digests" "MERAK_FORCE_SCALAR_ADAMW=1"
        "CKPT_INCREMENTAL_REQUIRE_1C1S=0" "CKPT_RUNTIME_ONLY=1"
    )
    if [[ "${mode}" != reference ]]; then
        envs+=("CKPT_REPLICA_ENDPOINTS=[\"172.20.17.2:${service_port}\",\"172.20.8.2:${service_port}\"]"
            "CKPT_OBJECT_RUN_ID=$(basename "${output_root}")" "CKPT_OBJECT_ROOT=${output_root}/source"
            "CKPT_OBJECT_DETECTOR_CONFIG=${output_root}/detector-experiment-config.json"
            "CKPT_OBJECT_MAX_INFLIGHT=${max_inflight}"
            "CKPT_OBJECT_BASE_UPLOAD_PARALLELISM=${base_upload_parallelism}"
            "CKPT_OBJECT_BASE_MODE=${base_mode}"
            "CKPT_OBJECT_SERVICE_TIMEOUT_SECONDS=${service_timeout_seconds}")
    fi
    if [[ "${mode}" == source ]]; then envs+=("CKPT_OBJECT_SEMANTIC_ENABLE=1");
    elif [[ "${mode}" == restore ]]; then
        envs+=("CKPT_OBJECT_RECOVERY_FACTORY=${restore_factory}" "CKPT_OBJECT_RESTORE_EPOCH=${restore_epoch}" "CKPT_OBJECT_BASE_EPOCH=0" "CKPT_OBJECT_SURVIVING_HOST=$([[ "${failed_service_node}" == 17 ]] && printf 1 || printf 0)")
    fi
    local -a command=(env -u CKPT_INCREMENTAL_ENABLE -u CKPT_INCREMENTAL_RESTORE_EPOCH -u CKPT_REMOTE_RESTORE_EPOCH
        -u CKPT_OBJECT_SEMANTIC_ENABLE -u CKPT_OBJECT_RECOVERY_FACTORY -u CKPT_OBJECT_RESTORE_EPOCH
        -u CKPT_OBJECT_BASE_EPOCH -u CKPT_OBJECT_SURVIVING_HOST -u CKPT_OBJECT_BASE_UPLOAD_PARALLELISM -u CKPT_OBJECT_BASE_MODE "${envs[@]}"
        NODE_IDS=17,8 NPROC_PER_NODE=8 MASTER_PORT="${master_port}" RUN_ID="object-semantic-${phase}-$(basename "${output_root}")" LOG_DIR="${output_root}/logs/${phase}"
        tools/launch_ckpt_grouping.sh examples/checkpoint_grouping/run_workload.py
        --model "${model}" --pp "${pp}" --tp "${tp}" --dp "${dp}" --mesh-order "${mesh_order}" --output_dir "${root}/trainer-output"
        --per_device_train_batch_size 1 --gradient_accumulation_steps "${gradient_accumulation_steps}" --max_steps "${max_steps}" --num_train_epochs 1
        --dataloader_num_workers 0 --save false --logging_steps 1 --report_to none --disable_tqdm true --no_tie_modules true
        --seq_length "${sequence_length}" --seed 20260913 --bf16 true --adafactor false --lr_scheduler_type constant
        --learning_rate "${learning_rate}" --weight_decay "${weight_decay}"
        --adam_beta1 "${adam_beta1}" --adam_beta2 "${adam_beta2}" --adam_epsilon "${adam_epsilon}")
    printf '%q ' "${command[@]}" > "${output_root}/${phase}-command.txt"; printf '\n' >> "${output_root}/${phase}-command.txt"
    local remote_group_file="${output_root}/services/${phase}-node8.pgid"
    local remote_inner remote_outer quoted_command
    printf -v quoted_command ' %q' "${command[@]}"
    printf -v remote_inner 'printf "%%s\\n" "$$" > %q; exec env NODE_RANK=1%s' \
        "${remote_group_file}" "${quoted_command}"
    # --wait keeps SSH attached to the detached process group's actual exit code.
    printf -v remote_outer 'cd %q && exec setsid --wait bash -c %q' "${repo_root}" "${remote_inner}"
    local started_ns finished_ns
    started_ns="$(monotonic_ns)"
    ssh "${ssh_options[@]}" "${remote_host}" "${remote_outer}" > "${output_root}/logs/${phase}-node8-outer.log" 2>&1 &
    local remote_pid=$!
    setsid env NODE_RANK=0 "${command[@]}" > "${output_root}/logs/${phase}-node17-outer.log" 2>&1 &
    local local_pid=$! local_rc remote_rc
    active_local_group="${local_pid}"
    active_local_pid="${local_pid}"
    active_remote_ssh_pid="${remote_pid}"
    for _ in {1..20}; do [[ -s "${remote_group_file}" ]] && break; sleep 0.1; done
    [[ -s "${remote_group_file}" ]] && active_remote_group="$(<"${remote_group_file}")"
    set +e; wait "${local_pid}"; local_rc=$?; wait "${remote_pid}"; remote_rc=$?; set -e
    finished_ns="$(monotonic_ns)"
    active_local_group=""
    active_remote_group=""
    active_local_pid=""
    active_remote_ssh_pid=""
    "${python}" - "${output_root}/launcher-status.json" "${phase}" "${local_rc}" "${remote_rc}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
phase, local, remote = sys.argv[2:]
value = json.loads(path.read_text()) if path.exists() else {}
value[phase] = {"local": int(local), "remote": int(remote)}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
    "${python}" - "${output_root}/phase-times.json" "${phase}" "${started_ns}" "${finished_ns}" "${source_steps}" "${continue_steps}" "${reference_steps}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
phase, started, finished, source, continued, reference = sys.argv[2:]
source, continued, reference = int(source), int(continued), int(reference)
value = json.loads(path.read_text()) if path.exists() else {}
scope = (
    "synchronized base, ten training steps, and final drain"
    if phase == "source" and source == 10
    else "fresh construction, restore to step 10, and five continued steps"
    if phase == "restore" and source == 10 and continued == 5
    else "uninterrupted fifteen-step no-checkpoint reference"
    if phase == "reference" and reference == 15
    else f"synchronized base, {source} training steps, and final drain"
    if phase == "source"
    else f"fresh construction, restore to step {source}, and {continued} continued steps"
    if phase == "restore"
    else f"uninterrupted {reference}-step no-checkpoint reference"
)
value[phase] = {
    "started_monotonic_ns": int(started), "finished_monotonic_ns": int(finished),
    "duration_ns": int(finished) - int(started),
    "scope": scope,
}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
    (( local_rc == 0 && remote_rc == 0 ))
}

if [[ "${arm}" == nockpt ]]; then
    run_phase reference "${reference_steps}" reference
else
    run_phase source "${source_steps}" source
    service_stats "${output_root}/source-service-after.json"
fi
if [[ "${arm}" != nockpt && "${restore_enabled}" == 1 ]]; then
    if [[ "${failed_service_node}" == 17 ]]; then kill -TERM "${local_service_pid}"; local_service_pid=""; else ssh "${ssh_options[@]}" "${remote_host}" "kill -TERM ${remote_service_pid}"; remote_service_pid=""; fi
    sleep 1
    service_stats "${output_root}/restore-service-before.json"
    run_phase restore "${reference_steps}" restore "${source_steps}"
    service_stats "${output_root}/restore-service-after.json"
fi
"${python}" tools/analyze_object_semantic_gate.py "${output_root}" --output "${output_root}/analysis.json"
printf 'complete_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${output_root}/COMPLETE"
