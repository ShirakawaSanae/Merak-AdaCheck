"""Merak adapter for ordered, one-update-per-step checkpoint logs."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import inspect
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

from .. import mpu
from ..amp.amp_optimizer import HalfPrecisionOptimizer
from ..recompute import get_rng_tracker
from .catalog import checkpoint_state_checksum, tensor_checksum
from .adacheck_semantic import (
    AdaCheckSemanticBinding,
    SemanticEncoderPlan,
    build_gradient_semantic_plan,
    semantic_binding_from_environment,
    semantic_binding_from_detector_config,
)
from .communication_profile import CommunicationProfile
from .grouping import grouping_config_from_env, plan_checkpoint_groups
from .global_commit import (
    FilesystemGlobalCommitCoordinator,
    GlobalCommitCoordinator,
    GlobalCommitRepository,
    manifest_evidence,
    rank_evidence,
)
from .incremental_checkpoint import RecoveryBinding, apply_optimizer_update
from .incremental_runtime import (
    IncrementalCheckpointSession,
    PeerIncrementalCheckpointSession,
    restore_incremental_chain,
    restore_own_materialized,
    restore_peer_incremental_chain,
    restore_peer_materialized,
)
from .recovery import ServiceRemoteCheckpointSession
from .state_oracle import compare_checkpoint_state


@contextmanager
def _node_materialized_read_lock(run_id: str):
    """Bound concurrent full-state readers on each physical host."""
    try:
        readers = int(os.environ.get("CKPT_MATERIALIZED_READERS_PER_NODE", "1"))
    except ValueError as error:
        raise ValueError("materialized readers per node must be 1 or 2") from error
    if readers not in {1, 2}:
        raise ValueError("materialized readers per node must be 1 or 2")
    timeout = float(
        os.environ.get("CKPT_MATERIALIZED_READ_LOCK_TIMEOUT_SECONDS", "3600")
    )
    if timeout <= 0:
        raise ValueError("materialized read lock timeout must be positive")
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    root = Path(os.environ.get("CKPT_MATERIALIZED_LOCK_ROOT", "/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        root / f"merak-materialized-read-{os.getuid()}-{digest}-{slot}.lock"
        for slot in range(readers)
    ]
    descriptors = [os.open(path, os.O_CREAT | os.O_RDWR, 0o600) for path in paths]
    start_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    order = [((local_rank + offset) % readers) for offset in range(readers)]
    acquired = None
    try:
        while acquired is None:
            for slot in order:
                try:
                    fcntl.flock(
                        descriptors[slot], fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    acquired = slot
                    break
                except BlockingIOError:
                    continue
            if acquired is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"materialized reader lock exceeded {timeout:g} seconds"
                )
            time.sleep(0.05)
        yield {
            "slot": acquired,
            "start_ns": start_ns,
            "acquired_ns": time.monotonic_ns(),
        }
    finally:
        if acquired is not None:
            fcntl.flock(descriptors[acquired], fcntl.LOCK_UN)
        for descriptor in descriptors:
            os.close(descriptor)


def incremental_checkpointing_enabled() -> bool:
    return os.environ.get("CKPT_INCREMENTAL_ENABLE", "0") == "1"


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def capture_runtime_state(epoch: int, sampler: Any = None) -> dict[str, Any]:
    state = {
        "iteration": int(epoch),
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "rng_tracker_states": get_rng_tracker().get_states(),
    }
    if sampler is not None:
        state["sampler"] = {
            name: int(getattr(sampler, name))
            for name in ("consumed_samples", "epoch")
            if hasattr(sampler, name)
        }
    return _cpu_snapshot(state)


def restore_runtime_state(state: Mapping[str, Any], sampler: Any = None) -> int:
    required = {
        "iteration",
        "random_rng_state",
        "np_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
    }
    if not required.issubset(state):
        raise ValueError("incremental runtime state is incomplete")
    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"])
    get_rng_tracker().set_states(state["rng_tracker_states"])
    if sampler is not None:
        for name, value in state.get("sampler", {}).items():
            if name not in {"consumed_samples", "epoch"} or not hasattr(sampler, name):
                raise ValueError(f"unsupported sampler state field: {name}")
            setattr(sampler, name, int(value))
    return int(state["iteration"])


def half_precision_parameter_ids(
    module: torch.nn.Module, optimizer: HalfPrecisionOptimizer
) -> tuple[tuple[str, ...], ...]:
    """Map FP32 master parameters to stable local model names."""
    aliases: dict[int, list[str]] = {}
    try:
        named = module.named_parameters(remove_duplicate=False)
    except TypeError:
        named = module.named_parameters()
    for name, parameter in named:
        aliases.setdefault(id(parameter), []).append(name)

    groups = []
    seen = set()
    for model_group, master_group, inner_group in zip(
        optimizer.fp16_groups,
        optimizer.fp32_groups,
        optimizer.optimizer.param_groups,
    ):
        if len(model_group) != len(master_group) or any(
            current is not master
            for current, master in zip(inner_group["params"], master_group)
        ):
            raise ValueError("half-precision optimizer parameter groups are inconsistent")
        identifiers = []
        for model_parameter in model_group:
            names = aliases.get(id(model_parameter))
            if not names:
                raise ValueError("optimizer parameter is absent from the local model")
            if len(names) != 1:
                raise ValueError("incremental checkpointing does not support tied parameters")
            identifier = min(names)
            if identifier in seen:
                raise ValueError("optimizer parameter names are not unique")
            seen.add(identifier)
            identifiers.append(identifier)
        groups.append(tuple(identifiers))
    if len(groups) != len(optimizer.optimizer.param_groups):
        raise ValueError("half-precision optimizer group count changed")
    return tuple(groups)


def _rank_pairs(world_size: int, ranks_per_node: int) -> tuple[tuple[int, int], ...]:
    encoded = os.environ.get("CKPT_INCREMENTAL_GROUPS")
    path = os.environ.get("CKPT_INCREMENTAL_GROUPS_FILE")
    if encoded is not None and path is not None:
        raise ValueError("set only one of CKPT_INCREMENTAL_GROUPS and its FILE form")
    if path is not None:
        encoded = Path(path).read_text()
    if encoded is None:
        groups = plan_checkpoint_groups(
            world_size, grouping_config_from_env()
        ).groups
    else:
        groups = json.loads(encoded)
    pairs = tuple(tuple(int(rank) for rank in group) for group in groups)
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("incremental checkpoint groups must be rank pairs")
    flattened = [rank for pair in pairs for rank in pair]
    if sorted(flattened) != list(range(world_size)):
        raise ValueError("incremental checkpoint groups must cover every rank once")
    if any(first // ranks_per_node == second // ranks_per_node for first, second in pairs):
        raise ValueError("incremental checkpoint peers must cross physical hosts")
    return pairs  # type: ignore[return-value]


def _binding(rank: int, world_size: int) -> RecoveryBinding:
    return RecoveryBinding(
        rank=rank,
        world_size=world_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        pipeline_parallel_rank=mpu.get_pipe_parallel_rank(),
        tensor_parallel_rank=mpu.get_model_parallel_rank(),
    )


class MerakIncrementalCheckpointController:
    """Attach the shared incremental log to Merak's BF16 optimizer boundary."""

    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: HalfPrecisionOptimizer,
        scheduler: Any,
        *,
        mode: str,
        session: IncrementalCheckpointSession | PeerIncrementalCheckpointSession,
        binding: RecoveryBinding,
        peer_binding: RecoveryBinding,
        parameter_ids: Sequence[Sequence[str]],
        epoch_provider: Callable[[], int],
        runtime_capture: Callable[[int], Mapping[str, Any]],
        runtime_restore: Callable[[Mapping[str, Any]], int],
        metrics_path: Path | None = None,
        materialized: bool = False,
        sealed: bool = False,
        materialized_restore_source: str = "peer",
        run_id: str = "test",
        global_commit_root: Path | None = None,
        max_lag_steps: int = 1,
        semantic_binding: AdaCheckSemanticBinding | None = None,
        semantic_plan: SemanticEncoderPlan | None = None,
        semantic_detector_evidence: Mapping[str, Any] | None = None,
        communication_profile: CommunicationProfile | None = None,
        global_commit_protocol: str = "collective-v1",
    ) -> None:
        if mode not in {"full_remote", "peer_delta"}:
            raise ValueError("incremental mode must be full_remote or peer_delta")
        if max_lag_steps < 1:
            raise ValueError("maximum checkpoint lag must be positive")
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode
        self.session = session
        self.binding = binding
        self.peer_binding = peer_binding
        self.parameter_ids = tuple(tuple(group) for group in parameter_ids)
        self.epoch_provider = epoch_provider
        self.runtime_capture = runtime_capture
        self.runtime_restore = runtime_restore
        self.metrics_path = metrics_path
        self.materialized = materialized
        if sealed and not materialized:
            raise ValueError("sealed checkpoints require materialization")
        if sealed and global_commit_root is None:
            raise ValueError("sealed checkpoints require global commit control")
        if materialized_restore_source not in {"peer", "own"}:
            raise ValueError("materialized restore source must be peer or own")
        self.sealed = sealed
        self.materialized_restore_source = materialized_restore_source
        self.run_id = run_id
        self.max_lag_steps = max_lag_steps
        self.semantic_binding = semantic_binding
        self.semantic_plan = semantic_plan
        self.semantic_detector_evidence = (
            dict(semantic_detector_evidence)
            if semantic_detector_evidence is not None
            else None
        )
        if semantic_plan is not None:
            if semantic_binding is None:
                raise ValueError("SemanticFixed requires a bound checkpoint stream")
            if (
                semantic_plan.policy != semantic_binding.policy
                or semantic_plan.catalog_hash != semantic_binding.catalog_hash
                or semantic_plan.parallel_plan_hash
                != semantic_binding.parallel_plan_hash
                or semantic_plan.protection_plan_hash
                != semantic_binding.protection_plan_hash
                or semantic_plan.world_size != binding.world_size
            ):
                raise ValueError("semantic encoder plan and binding disagree")
        self.communication_profile = communication_profile
        if global_commit_protocol not in {
            "collective-v1",
            FilesystemGlobalCommitCoordinator.protocol,
        }:
            raise ValueError("unsupported global commit coordinator")
        self.global_commit_protocol = global_commit_protocol
        self.global_repository = (
            GlobalCommitRepository(global_commit_root)
            if global_commit_root is not None
            else None
        )
        self.global_coordinator: (
            GlobalCommitCoordinator | FilesystemGlobalCommitCoordinator | None
        ) = None
        self.base_epoch: int | None = None
        self.base_commit: Any = None
        self.captured_epochs: list[int] = []
        self.committed: dict[int, Any] = {}
        self.capture_ms: list[float] = []
        self.boundary_lag: list[int] = []
        self.global_boundary_lag: list[int] = []
        self.base_buffer_evidence: tuple[tuple[str, str], ...] | None = None
        self.base_equivalence_checksum: str | None = None

    def _semantic_encoder_evidence(self) -> dict[str, Any] | None:
        if self.semantic_plan is None:
            return None
        rank_actions = self.semantic_plan.actions_for_rank(self.binding.rank)
        evidence = {
            "schema_version": 1,
            "kind": "adacheck_semantic_fixed",
            "plan_hash": self.semantic_plan.plan_hash,
            "trainable_action": "gradient_bf16",
            "gradient_dtype": "torch.bfloat16",
            "gradient_snapshot_point": "raw_bf16_pre_unscale",
            "gradient_transform": "upcast_fp32_times_exact_step_scale",
            "metadata_action": "metadata_exact",
            "rank_action_count": len(rank_actions),
            "global_action_count": len(self.semantic_plan.actions),
        }
        if self.mode == "peer_delta":
            evidence.update(
                {
                    "protection_action": "fixed_dp_peer_reuse",
                    "remote_trainable_payload": "none",
                    "peer_equivalence": "per_epoch_sha256",
                }
            )
        else:
            evidence.update(
                {
                    "protection_action": "remote_gradient",
                    "remote_trainable_payload": "gradient_bf16",
                    "peer_equivalence": None,
                }
            )
        return evidence

    def record_profile_interval(
        self, name: str, start_ns: int, end_ns: int, **fields: Any
    ) -> None:
        if self.communication_profile is not None:
            self.communication_profile.record_interval(
                name,
                start_ns,
                end_ns,
                rank=self.binding.rank,
                **fields,
            )

    def _validate_semantic_encoder_value(self, value: Any) -> None:
        expected = self._semantic_encoder_evidence()
        if expected is None:
            return
        if not isinstance(value, Mapping) or dict(value) != expected:
            raise ValueError("SemanticFixed encoder plan mismatch")

    def _buffer_evidence(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, tensor_checksum(buffer))
            for name, buffer in self.module.named_buffers()
        )

    def _materialized_rank_barrier(self, phase: str) -> None:
        """Keep long host-serialized recovery work outside NCCL collectives."""
        if self.global_repository is None or not dist.is_initialized():
            return
        root = self.global_repository.root / "rank-barriers" / phase
        root.mkdir(parents=True, exist_ok=True)
        ready = root / f"rank_{self.binding.rank:05d}.ready"
        temporary = ready.with_name(f".{ready.name}.{os.getpid()}.tmp")
        temporary.write_text(f"{self.binding.rank}\n")
        temporary.replace(ready)
        timeout = float(
            os.environ.get("CKPT_MATERIALIZED_BARRIER_TIMEOUT_SECONDS", "3600")
        )
        deadline = time.monotonic() + timeout
        while len(tuple(root.glob("rank_*.ready"))) < self.binding.world_size:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"materialized {phase} rank barrier exceeded {timeout:g} seconds"
                )
            time.sleep(0.1)

    def _state_components(self) -> dict[str, Any]:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
        }

    def _record_state_checksums(self, environment_key: str, epoch: int) -> None:
        root = os.environ.get(environment_key)
        if not root:
            return
        components = self._state_components()
        value = {
            "schema_version": 2,
            "rank": self.binding.rank,
            "iteration": int(epoch),
            "state_checksum": checkpoint_state_checksum(components),
            "component_checksums": {
                name: checkpoint_state_checksum(state)
                for name, state in components.items()
            },
        }
        destination = Path(root) / f"rank_{self.binding.rank:05d}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)

    def _materialized_state_oracle(self, epoch: int) -> dict[str, Any]:
        with _node_materialized_read_lock(self.run_id) as lock:
            self.record_profile_interval(
                "checkpoint_materialized_reader_wait",
                lock["start_ns"],
                lock["acquired_ns"],
                epoch=epoch,
                checkpoint_mode=self.mode,
                reader_slot=lock["slot"],
                operation="validate",
            )
            return self._materialized_state_oracle_unlocked(epoch)

    def _materialized_state_oracle_unlocked(self, epoch: int) -> dict[str, Any]:
        assert self.global_repository is not None
        control = self.global_repository.load(
            run_id=self.run_id,
            epoch=epoch,
            world_size=self.binding.world_size,
        )
        rank = rank_evidence(control, self.binding.rank)
        full_session = (
            self.session.session
            if self.mode == "full_remote"
            else self.session.local_full_session
        )
        path = "remote_full" if self.mode == "full_remote" else "local_full"
        fetch_started_ns = time.monotonic_ns()
        materialized, _ = restore_own_materialized(
            full_session.client,
            rank=self.binding.rank,
            epoch=epoch,
            group_ranks=full_session.group_ranks,
            run_id=self.run_id,
            expected_full_evidence=rank["paths"][path],
            control_checksum=str(control["control_checksum"]),
            expected_target_rank=(
                self.peer_binding.rank if self.mode == "full_remote" else None
            ),
        )
        self.record_profile_interval(
            "checkpoint_materialized_fetch_deserialize",
            fetch_started_ns,
            time.monotonic_ns(),
            epoch=epoch,
            checkpoint_mode=self.mode,
            operation="validate",
        )
        compare_started_ns = time.monotonic_ns()
        actual = {
            "model": materialized["model"],
            "optimizer": materialized["optimizer"],
        }
        expected = self._state_components()
        components = {
            name: compare_checkpoint_state(actual[name], expected[name])
            for name in actual
        }
        result = {
            "epoch": epoch,
            "components": components,
            "bitwise_equal": {
                name: result["bitwise_equal"]
                for name, result in components.items()
            },
            "numerically_close": {
                name: result["numerically_close"]
                for name, result in components.items()
            },
            "all_bitwise_equal": all(
                result["bitwise_equal"] for result in components.values()
            ),
            "all_numerically_close": all(
                result["numerically_close"] for result in components.values()
            ),
        }
        self.record_profile_interval(
            "checkpoint_materialized_oracle_compare",
            compare_started_ns,
            time.monotonic_ns(),
            epoch=epoch,
            checkpoint_mode=self.mode,
        )
        return result

    def _base_state(self, epoch: int) -> dict[str, Any]:
        state = {
            "schema_version": 1,
            "kind": "merak_incremental_base",
            "iteration": epoch,
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
            "lr_scheduler_type": (
                f"{type(self.scheduler).__module__}."
                f"{type(self.scheduler).__qualname__}"
            ),
            "parameter_ids": [list(group) for group in self.parameter_ids],
            "binding": asdict(self.binding),
            "run_id": self.run_id,
        }
        if self.materialized:
            state["stream_role"] = (
                "remote_full" if self.mode == "full_remote" else "peer_local_full"
            )
        if self.semantic_binding is not None:
            state["adacheck_semantic"] = self.semantic_binding.to_dict()
        semantic_encoder = self._semantic_encoder_evidence()
        if semantic_encoder is not None:
            state["adacheck_semantic_encoder"] = semantic_encoder
            state["adacheck_detector_evidence"] = self.semantic_detector_evidence
        return state

    def start(self, epoch: int) -> None:
        if self.base_epoch is not None:
            raise RuntimeError("incremental controller is already started")
        if self.optimizer.global_step != epoch:
            raise ValueError("optimizer and trainer step counters disagree at the base")
        runtime = self.runtime_capture(epoch)
        state = self._base_state(epoch)
        state["runtime_state"] = runtime
        if self.mode == "full_remote":
            self.base_commit = self.session.save_base(epoch, state)
        else:
            full, delta = self.session.save_base(
                epoch, state, runtime_state=runtime
            )
            self.base_commit = (full, delta)
            evidence = [
                (item.path, item.dtype, item.shape, item.checksum)
                for item in full.manifest.objects
            ]
            digest = hashlib.sha256(
                json.dumps(evidence, separators=(",", ":")).encode()
            ).hexdigest()
            if dist.is_initialized():
                digests: list[Any] = [None] * dist.get_world_size()
                dist.all_gather_object(digests, digest)
                if digests[self.peer_binding.rank] != digest:
                    raise ValueError("incremental peer base tensors are not identical")
            self.base_equivalence_checksum = digest
        self.base_epoch = epoch
        if self.global_repository is not None:
            if not dist.is_initialized():
                raise RuntimeError("global commit requires torch.distributed")
            coordinator_class = (
                FilesystemGlobalCommitCoordinator
                if self.global_commit_protocol
                == FilesystemGlobalCommitCoordinator.protocol
                else GlobalCommitCoordinator
            )
            self.global_coordinator = coordinator_class(
                rank=self.binding.rank,
                world_size=self.binding.world_size,
                run_id=self.run_id,
                root=self.global_repository.root,
                initial_epoch=epoch - 1,
                before_publish=(self.session.pin_epoch if self.sealed else None),
                after_publish=(self.session.seal_epoch if self.sealed else None),
            )
            self.global_coordinator.submit_epoch(
                epoch, self._rank_commit_evidence(epoch)
            )
            if isinstance(
                self.global_coordinator, FilesystemGlobalCommitCoordinator
            ):
                self.global_coordinator.wait_through(epoch)
        self.base_buffer_evidence = self._buffer_evidence()
        self.optimizer.set_pre_optimizer_step_callback(
            self._capture_update,
            capture_raw_gradients=self.semantic_plan is not None,
        )

    def _capture_update(
        self,
        optimizer_epoch: int,
        inner_optimizer: torch.optim.Optimizer,
        gradient_multiplier: float,
    ) -> None:
        capture_started_ns = time.monotonic_ns()
        epoch = int(self.epoch_provider())
        if inner_optimizer is not self.optimizer.optimizer or optimizer_epoch != epoch:
            raise ValueError("optimizer callback does not match the trainer step")
        expected = (self.base_epoch or 0) + len(self.captured_epochs) + 1
        if epoch != expected:
            raise ValueError(f"incremental update epoch {epoch} is not {expected}")
        runtime = self.runtime_capture(epoch)
        metadata = {
            "trainer_global_step": epoch,
            "scheduler_state_before_step": _cpu_snapshot(
                self.scheduler.state_dict()
            ),
        }
        if self.materialized:
            metadata["stream_role"] = (
                "remote_full" if self.mode == "full_remote" else "peer_local_full"
            )
        metadata["run_id"] = self.run_id
        if self.semantic_binding is not None:
            metadata["adacheck_semantic"] = self.semantic_binding.to_dict()
        semantic_encoder = self._semantic_encoder_evidence()
        if semantic_encoder is not None:
            metadata["adacheck_semantic_encoder"] = semantic_encoder
        launched = self.session.enqueue_update(
            epoch,
            metadata=metadata,
            runtime_state=runtime,
            gradient_multiplier=gradient_multiplier,
        )
        self.captured_epochs.append(epoch)
        self.capture_ms.append(
            float(sum(launched) if isinstance(launched, tuple) else launched)
        )
        self.record_profile_interval(
            "checkpoint_capture",
            capture_started_ns,
            time.monotonic_ns(),
            epoch=epoch,
            checkpoint_mode=self.mode,
        )

    def _accept(self, completed: Sequence[Any]) -> None:
        for item in completed:
            if item.epoch in self.committed:
                raise RuntimeError(f"incremental epoch {item.epoch} committed twice")
            self.committed[item.epoch] = item

    def poll(self, current_epoch: int) -> None:
        if self.base_epoch is None:
            return
        poll_started_ns = time.monotonic_ns()
        try:
            self._accept(self.session.poll())
            latest = max(self.committed, default=self.base_epoch)
            self.boundary_lag.append(int(current_epoch) - latest)
            global_latest = latest
            if self.global_coordinator is not None:
                global_latest = self.global_coordinator.advance(
                    latest, self._rank_commit_evidence
                )
            self.global_boundary_lag.append(int(current_epoch) - global_latest)
        finally:
            self.record_profile_interval(
                "checkpoint_poll",
                poll_started_ns,
                time.monotonic_ns(),
                epoch=int(current_epoch),
                checkpoint_mode=self.mode,
            )

    def _rank_commit_evidence(self, epoch: int) -> dict[str, Any]:
        item = self.base_commit if epoch == self.base_epoch else self.committed[epoch]
        if self.mode == "full_remote":
            result = item if epoch == self.base_epoch else item.result
            paths = {"remote_full": manifest_evidence(result.manifest)}
        elif epoch == self.base_epoch:
            full, delta = item
            paths = {
                "local_full": manifest_evidence(full.manifest),
                "remote_delta": manifest_evidence(delta.manifest),
            }
        else:
            paths = {
                "local_full": manifest_evidence(item.local_full.result.manifest),
                "remote_delta": manifest_evidence(item.remote_delta.result.manifest),
            }
        evidence = {
            "rank": self.binding.rank,
            "epoch": epoch,
            "mode": self.mode,
            "paths": paths,
        }
        if self.sealed and self.mode == "peer_delta":
            checksum = (
                self.base_equivalence_checksum
                if epoch == self.base_epoch
                else item.local_full.result.equivalence_checksum
            )
            if checksum is None:
                raise RuntimeError("sealed peer update lacks equivalence evidence")
            evidence["peer_equivalence_checksum"] = checksum
        if self.semantic_binding is not None:
            evidence["adacheck_semantic"] = self.semantic_binding.to_dict()
        semantic_encoder = self._semantic_encoder_evidence()
        if semantic_encoder is not None:
            evidence["adacheck_semantic_encoder"] = semantic_encoder
        return evidence

    def finalize(self, current_epoch: int) -> dict[str, Any]:
        if self.base_epoch is None:
            return {}
        finalize_started_ns = time.monotonic_ns()
        self.optimizer.set_pre_optimizer_step_callback(None)
        publisher_started_ns = time.monotonic_ns()
        self._accept(self.session.close())
        self.record_profile_interval(
            "checkpoint_publisher_drain",
            publisher_started_ns,
            time.monotonic_ns(),
            epoch=int(current_epoch),
            checkpoint_mode=self.mode,
        )
        global_latest = max(self.committed, default=self.base_epoch)
        global_started_ns = time.monotonic_ns()
        if self.global_coordinator is not None:
            global_latest = self.global_coordinator.advance(
                global_latest, self._rank_commit_evidence
            )
            global_latest = self.global_coordinator.close()
            if self.sealed:
                self.session.wait_sealed(global_latest)
        self.record_profile_interval(
            "checkpoint_global_publish_seal",
            global_started_ns,
            time.monotonic_ns(),
            epoch=int(current_epoch),
            checkpoint_mode=self.mode,
        )
        materialized_oracle = None
        if (
            self.sealed
            and self.materialized
            and os.environ.get("CKPT_VALIDATE_MATERIALIZED_STATE", "0") == "1"
        ):
            materialized_oracle = self._materialized_state_oracle(global_latest)
        expected = list(range(self.base_epoch + 1, int(current_epoch) + 1))
        committed = sorted(self.committed)
        commit_metrics = []
        for epoch in committed:
            item = self.committed[epoch]
            if self.mode == "full_remote":
                commit_metrics.append(self._commit_metrics("remote_full", item))
            else:
                commit_metrics.extend(
                    (
                        self._commit_metrics("local_full", item.local_full),
                        self._commit_metrics("remote_delta", item.remote_delta),
                    )
                )
        summary = {
            "schema_version": 2,
            "run_kind": "merak_incremental_rank",
            "rank": self.binding.rank,
            "mode": self.mode,
            "base_epoch": self.base_epoch,
            "final_epoch": int(current_epoch),
            "captured_epochs": self.captured_epochs,
            "committed_epochs": committed,
            "all_epochs_contiguous": self.captured_epochs == expected
            and committed == expected,
            "max_boundary_lag_steps": max(self.boundary_lag, default=0),
            "boundary_lag_steps": self.boundary_lag,
            "configured_max_lag_steps": self.max_lag_steps,
            "global_committed_epoch": global_latest,
            "global_committed_epochs": list(
                range(self.base_epoch + 1, global_latest + 1)
            ),
            "max_global_boundary_lag_steps": max(
                self.global_boundary_lag, default=0
            ),
            "global_boundary_lag_steps": self.global_boundary_lag,
            "commit_scope": (
                "global_manifest"
                if self.global_coordinator is not None
                else "rank_ack_only"
            ),
            "global_commit_protocol": (
                self.global_coordinator.protocol
                if self.global_coordinator is not None
                else None
            ),
            "sealed": self.sealed,
            "publication_protocol": (
                "pin-before-publish-v1" if self.sealed else None
            ),
            "peer_equivalence": (
                "per_epoch_sha256"
                if self.sealed and self.mode == "peer_delta"
                else None
            ),
            "materialized_state_oracle": materialized_oracle,
            "capture_ms": self.capture_ms,
            "commits": commit_metrics,
        }
        semantic_encoder = self._semantic_encoder_evidence()
        if semantic_encoder is not None:
            summary["adacheck_semantic_encoder"] = semantic_encoder
            summary["adacheck_detector_evidence"] = self.semantic_detector_evidence
        summary["model_buffers_unchanged"] = (
            self._buffer_evidence() == self.base_buffer_evidence
        )
        summary["recovery_valid"] = bool(
            summary["all_epochs_contiguous"]
            and global_latest == int(current_epoch)
            and summary["model_buffers_unchanged"]
            and (
                materialized_oracle is None
                or materialized_oracle["all_numerically_close"]
            )
        )
        summary["cadence_valid"] = bool(
            summary["max_boundary_lag_steps"] <= self.max_lag_steps
            and summary["max_global_boundary_lag_steps"] <= self.max_lag_steps
        )
        summary["strict_cadence_required"] = (
            os.environ.get("CKPT_INCREMENTAL_REQUIRE_1C1S", "1") == "1"
        )
        summary["valid"] = bool(
            summary["recovery_valid"]
            and (
                summary["cadence_valid"]
                or not summary["strict_cadence_required"]
            )
        )
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.metrics_path.with_name(f".{self.metrics_path.name}.tmp")
            temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            temporary.replace(self.metrics_path)
        if materialized_oracle is not None:
            barrier_started_ns = time.monotonic_ns()
            self._materialized_rank_barrier(f"validate-epoch-{global_latest:08d}")
            self.record_profile_interval(
                "checkpoint_materialized_validation_barrier",
                barrier_started_ns,
                time.monotonic_ns(),
                epoch=int(current_epoch),
                checkpoint_mode=self.mode,
            )
        finalize_finished_ns = time.monotonic_ns()
        self.record_profile_interval(
            "checkpoint_final_drain",
            finalize_started_ns,
            finalize_finished_ns,
            epoch=int(current_epoch),
            checkpoint_mode=self.mode,
        )
        self.record_profile_interval(
            "checkpoint_finalize_total",
            finalize_started_ns,
            finalize_finished_ns,
            epoch=int(current_epoch),
            checkpoint_mode=self.mode,
        )
        if self.communication_profile is not None:
            self.communication_profile.close()
        return summary

    @staticmethod
    def _commit_metrics(path: str, item: Any) -> dict[str, Any]:
        result = item.result
        metrics = {
            "epoch": item.epoch,
            "path": path,
            "capture_ms": item.capture_ms,
            "queue_ms": item.queue_ms,
            "publisher_queue_ms": item.publisher_queue_ms,
            "snapshot_ready_wait_ms": item.snapshot_ready_wait_ms,
            "serialize_ms": result.serialize_ms,
            "shared_slot_wait_ms": result.shared_slot_wait_ms,
            "shared_copy_ms": result.shared_copy_ms,
            "transport_ms": result.transport.duration_ms,
            "commit_ms": result.commit_ms,
            "enqueue_to_commit_ms": item.enqueue_to_commit_ms,
            "put_service_ms": result.put_service_ms,
            "commit_service_ms": result.commit_service_ms,
            "payload_bytes": result.transport.payload_bytes,
            "wire_bytes": result.transport.wire_bytes,
            "logical_bytes": result.manifest.logical_bytes,
            "transport_kind": result.transport_kind,
            "shared_memory_bytes": result.shared_memory_bytes,
        }
        if result.transport_kind != "shared_tensor":
            metrics.pop("shared_slot_wait_ms")
            metrics.pop("shared_copy_ms")
        return metrics

    def restore(self, *, base_epoch: int, final_epoch: int) -> int:
        restore_started_ns = time.monotonic_ns()
        if self.materialized:
            with _node_materialized_read_lock(self.run_id) as lock:
                self.record_profile_interval(
                    "checkpoint_materialized_reader_wait",
                    lock["start_ns"],
                    lock["acquired_ns"],
                    epoch=final_epoch,
                    checkpoint_mode=self.mode,
                    reader_slot=lock["slot"],
                    operation="restore",
                )
                iteration = self._restore_unlocked(
                    base_epoch=base_epoch, final_epoch=final_epoch
                )
            barrier_started_ns = time.monotonic_ns()
            self._materialized_rank_barrier(f"restore-epoch-{final_epoch:08d}")
            self.record_profile_interval(
                "checkpoint_materialized_restore_barrier",
                barrier_started_ns,
                time.monotonic_ns(),
                epoch=final_epoch,
                checkpoint_mode=self.mode,
            )
            self.record_profile_interval(
                "checkpoint_restore_total",
                restore_started_ns,
                time.monotonic_ns(),
                epoch=final_epoch,
                checkpoint_mode=self.mode,
            )
            return iteration
        iteration = self._restore_unlocked(base_epoch=base_epoch, final_epoch=final_epoch)
        self.record_profile_interval(
            "checkpoint_restore_total",
            restore_started_ns,
            time.monotonic_ns(),
            epoch=final_epoch,
            checkpoint_mode=self.mode,
        )
        return iteration

    def _restore_unlocked(self, *, base_epoch: int, final_epoch: int) -> int:
        if self.materialized and (
            self.global_repository is None or self.run_id == "test"
        ):
            raise RuntimeError(
                "materialized restore requires global commit evidence and a run ID"
            )
        control = None
        if self.global_repository is not None:
            control = self.global_repository.load(
                run_id=self.run_id,
                epoch=final_epoch,
                world_size=self.binding.world_size,
            )
            if self.sealed and control.get("publication_protocol") != (
                "pin-before-publish-v1"
            ):
                raise ValueError("materialized restore requires a sealed global control")
            if self.semantic_binding is not None and control.get("adacheck_semantic") != (
                self.semantic_binding.to_dict()
            ):
                raise ValueError("global commit has incompatible AdaCheck semantic binding")
            semantic_encoder = self._semantic_encoder_evidence()
            if semantic_encoder is not None:
                expected_global_encoder = {
                    key: value
                    for key, value in semantic_encoder.items()
                    if key != "rank_action_count"
                }
                if control.get("adacheck_semantic_encoder") != expected_global_encoder:
                    raise ValueError("global commit has incompatible SemanticFixed plan")
        if self.materialized:
            failed_control = rank_evidence(control, self.binding.rank) if control else None
            peer_control = (
                rank_evidence(control, self.peer_binding.rank)
                if control and self.mode == "peer_delta"
                else None
            )
            fetch_started_ns = time.monotonic_ns()
            if self.mode == "full_remote":
                state, runtime = restore_own_materialized(
                    self.session.session.client,
                    rank=self.binding.rank,
                    epoch=final_epoch,
                    group_ranks=self.session.session.group_ranks,
                    run_id=self.run_id,
                    expected_full_evidence=failed_control["paths"]["remote_full"],
                    control_checksum=str(control["control_checksum"]),
                    expected_target_rank=self.peer_binding.rank,
                )
                materialized_binding = self.binding
                expected_source_evidence = failed_control["paths"]["remote_full"]
            elif self.materialized_restore_source == "own":
                state, runtime = restore_own_materialized(
                    self.session.local_full_session.client,
                    rank=self.binding.rank,
                    epoch=final_epoch,
                    group_ranks=self.session.local_full_session.group_ranks,
                    run_id=self.run_id,
                    expected_full_evidence=failed_control["paths"]["local_full"],
                    control_checksum=str(control["control_checksum"]),
                )
                materialized_binding = self.binding
                expected_source_evidence = failed_control["paths"]["local_full"]
            else:
                state, runtime = restore_peer_materialized(
                    self.session.remote_delta_session.client,
                    failed_rank=self.binding.rank,
                    peer_rank=self.peer_binding.rank,
                    epoch=final_epoch,
                    group_ranks=self.session.remote_delta_session.group_ranks,
                    run_id=self.run_id,
                    expected_full_evidence=peer_control["paths"]["local_full"],
                    expected_delta_evidence=failed_control["paths"]["remote_delta"],
                    control_checksum=str(control["control_checksum"]),
                )
                materialized_binding = self.peer_binding
                expected_source_evidence = peer_control["paths"]["local_full"]
            self.record_profile_interval(
                "checkpoint_materialized_fetch_deserialize",
                fetch_started_ns,
                time.monotonic_ns(),
                epoch=final_epoch,
                checkpoint_mode=self.mode,
                operation="restore",
            )
            load_started_ns = time.monotonic_ns()
            scheduler_type = (
                f"{type(self.scheduler).__module__}."
                f"{type(self.scheduler).__qualname__}"
            )
            materialization = state.get("materialization", {})
            if (
                state.get("parameter_ids")
                != [list(group) for group in self.parameter_ids]
                or state.get("binding") != asdict(materialized_binding)
                or state.get("lr_scheduler_type") != scheduler_type
                or state.get("run_id") != self.run_id
                or int(state["optimizer"].get("global_step", -1)) != final_epoch
                or int(runtime.get("iteration", -1)) != final_epoch
                or int(materialization.get("origin_base_epoch", -1)) != base_epoch
                or int(materialization.get("applied_updates", -1))
                != final_epoch - base_epoch
                or materialization.get("optimizer_replay") != "cpu_numerical"
                or materialization.get("source_manifest")
                != dict(expected_source_evidence)
            ):
                raise ValueError("materialized Merak state is incompatible")
            if self.semantic_binding is not None:
                self.semantic_binding.validate_base(state)
            self._validate_semantic_encoder_value(
                state.get("adacheck_semantic_encoder")
            )
            self.module.load_state_dict(state["model"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["lr_scheduler"])
            scheduler_steps = int(
                materialization.get("scheduler_steps_after_snapshot", -1)
            )
            if scheduler_steps not in {0, 1}:
                raise ValueError("materialized scheduler boundary is invalid")
            for _ in range(scheduler_steps):
                self.scheduler.step()
            self.optimizer.refresh_model_params()
            iteration = self.runtime_restore(runtime)
            self._record_state_checksums(
                "CKPT_RESTORE_BOUNDARY_DIGEST_DIR", iteration
            )
            self.record_profile_interval(
                "checkpoint_restore_validate_load",
                load_started_ns,
                time.monotonic_ns(),
                epoch=final_epoch,
                checkpoint_mode=self.mode,
            )
            return iteration
        if self.mode == "full_remote":
            chain = restore_incremental_chain(
                self.session.session,
                base_epoch=base_epoch,
                final_epoch=final_epoch,
            )
            expected_binding = self.binding
            runtime = (
                chain.base_state["runtime_state"]
                if not chain.updates
                else chain.updates[-1]["runtime_state"]
            )
        else:
            peer_chain = restore_peer_incremental_chain(
                self.session.remote_delta_session.client,
                failed_rank=self.binding.rank,
                peer_rank=self.peer_binding.rank,
                base_epoch=base_epoch,
                final_epoch=final_epoch,
            )
            chain = peer_chain.full
            expected_binding = self.peer_binding
            runtime = peer_chain.runtime_states[-1]
        base = chain.base_state
        if self.semantic_binding is not None:
            self.semantic_binding.validate_base(base)
            for update in chain.updates:
                self.semantic_binding.validate_update(update)
        self._validate_semantic_encoder_value(base.get("adacheck_semantic_encoder"))
        if self.semantic_plan is not None:
            for update in chain.updates:
                metadata = update.get("metadata")
                value = (
                    metadata.get("adacheck_semantic_encoder")
                    if isinstance(metadata, Mapping)
                    else None
                )
                self._validate_semantic_encoder_value(value)
        if (
            base.get("schema_version") != 1
            or base.get("kind") != "merak_incremental_base"
            or int(base.get("iteration", -1)) != base_epoch
            or base.get("parameter_ids")
            != [list(group) for group in self.parameter_ids]
        ):
            raise ValueError("Merak incremental base is incompatible with this rank")
        if int(base["optimizer"].get("global_step", -1)) != base_epoch:
            raise ValueError("incremental base optimizer epoch is inconsistent")
        if int(runtime.get("iteration", -1)) != final_epoch:
            raise ValueError("incremental runtime epoch is inconsistent")
        self.module.load_state_dict(base["model"], strict=True)
        self.optimizer.load_state_dict(base["optimizer"])
        self.scheduler.load_state_dict(base["lr_scheduler"])
        for epoch, update in enumerate(chain.updates, start=base_epoch + 1):
            apply_optimizer_update(
                self.optimizer.optimizer,
                update,
                expected_parameter_ids=self.parameter_ids,
                expected_binding=expected_binding,
                expected_base_id=chain.base_id,
                expected_base_epoch=base_epoch,
                expected_epoch=epoch,
            )
            self.scheduler.step()
        self.optimizer.global_step = final_epoch
        self.optimizer.cur_iter += len(chain.updates)
        self.optimizer.refresh_model_params()
        iteration = self.runtime_restore(runtime)
        self._record_state_checksums("CKPT_RESTORE_BOUNDARY_DIGEST_DIR", iteration)
        return iteration


def build_merak_incremental_controller(
    module: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    *,
    args: Any,
    sampler: Any,
    epoch_provider: Callable[[], int],
) -> MerakIncrementalCheckpointController:
    if not isinstance(optimizer, HalfPrecisionOptimizer) or not args.bf16:
        raise ValueError("Merak incremental checkpointing requires the BF16 optimizer")
    if args.zero_stage is not None:
        raise ValueError("Merak incremental checkpointing does not support ZeRO")
    if args.dataloader_num_workers != 0:
        raise ValueError(
            "incremental checkpointing requires dataloader_num_workers=0"
        )
    if type(optimizer.optimizer) is not torch.optim.AdamW:
        raise ValueError("Merak incremental checkpointing requires torch.optim.AdamW")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks_per_node = int(
        os.environ.get("CKPT_RANKS_PER_NODE", os.environ.get("LOCAL_WORLD_SIZE", "1"))
    )
    pairs = _rank_pairs(world_size, ranks_per_node)
    group = next(pair for pair in pairs if rank in pair)
    peer = group[1] if group[0] == rank else group[0]
    mode = os.environ.get("CKPT_INCREMENTAL_MODE", "peer_delta")
    binding = _binding(rank, world_size)
    bindings: list[Any] = [None] * world_size
    dist.all_gather_object(bindings, asdict(binding))
    peer_binding = RecoveryBinding(**bindings[peer])
    if mode == "peer_delta" and (
        binding.pipeline_parallel_rank != peer_binding.pipeline_parallel_rank
        or binding.tensor_parallel_rank != peer_binding.tensor_parallel_rank
        or binding.data_parallel_rank == peer_binding.data_parallel_rank
    ):
        raise ValueError("incremental peer must be a cross-host DP replica")

    parameter_ids = half_precision_parameter_ids(module, optimizer)
    if mode == "peer_delta":
        topologies: list[Any] = [None] * world_size
        dist.all_gather_object(topologies, parameter_ids)
        if tuple(tuple(group) for group in topologies[peer]) != parameter_ids:
            raise ValueError(
                "incremental peer has a different optimizer parameter topology"
            )

    endpoints = json.loads(os.environ["CKPT_REPLICA_ENDPOINTS"])
    host_count = world_size // ranks_per_node
    if not isinstance(endpoints, list) or len(endpoints) != host_count:
        raise ValueError("CKPT_REPLICA_ENDPOINTS must contain one endpoint per host")
    root = Path(
        os.environ.get(
            "CKPT_INCREMENTAL_MANIFEST_DIR",
            str(Path(args.output_dir) / "incremental_checkpoint"),
        )
    )
    run_id = os.environ.get("CKPT_INCREMENTAL_RUN_ID")
    if not run_id:
        raise ValueError("CKPT_INCREMENTAL_RUN_ID is required")
    common = {
        "rank": rank,
        "group_ranks": group,
        "max_local_store_bytes": 1,
        "collect_store_stats": False,
        "timeout_seconds": float(
            os.environ.get("CKPT_REPLICA_TIMEOUT_SECONDS", "60")
        ),
    }
    materialized = os.environ.get("CKPT_INCREMENTAL_MATERIALIZE", "0") == "1"
    sealed = os.environ.get("CKPT_INCREMENTAL_SEALED", "0") == "1"
    required_scheduler_parameters = [
        parameter
        for parameter in inspect.signature(scheduler.step).parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if materialized and required_scheduler_parameters:
        raise ValueError("CPU materialization requires a no-argument scheduler step")
    max_inflight = int(os.environ.get("CKPT_INCREMENTAL_MAX_INFLIGHT", "2"))
    semantic_encoder = os.environ.get("CKPT_ADACHECK_ENCODER")
    semantic_plan = None
    semantic_detector_evidence = None
    if semantic_encoder is None:
        semantic_binding = semantic_binding_from_environment()
    else:
        if semantic_encoder != "semantic_fixed_gradient_bf16":
            raise ValueError("unsupported AdaCheck semantic encoder")
        if mode not in {"full_remote", "peer_delta"} or host_count != 2:
            raise ValueError(
                "two-host SemanticFixed requires a supported checkpoint path"
            )
        detector_config = os.environ.get("CKPT_ADACHECK_DETECTOR_CONFIG")
        expected_model = os.environ.get("CKPT_ADACHECK_MODEL")
        if not detector_config or not expected_model:
            raise ValueError(
                "SemanticFixed requires a frozen detector config and model identity"
            )
        semantic_binding, semantic_detector_evidence = (
            semantic_binding_from_detector_config(
                Path(detector_config),
                expected_model=expected_model,
                world_size=world_size,
                ranks_per_host=ranks_per_node,
                rank_bindings=bindings,
                pairs=pairs,
            )
        )
        parameter_topologies: list[Any] = [None] * world_size
        dist.all_gather_object(parameter_topologies, parameter_ids)
        semantic_plan = build_gradient_semantic_plan(
            semantic_binding,
            world_size=world_size,
            pairs=pairs,
            parameter_ids_by_rank=parameter_topologies,
        )
        if rank == 0:
            semantic_root = root / "semantic-fixed"
            semantic_root.mkdir(parents=True, exist_ok=True)
            plan_path = semantic_root / "encoder-plan.json"
            plan_path.write_text(
                json.dumps(semantic_plan.to_dict(), indent=2, sort_keys=True) + "\n"
            )
            evidence_path = semantic_root / "detector-evidence.json"
            evidence_path.write_text(
                json.dumps(
                    semantic_detector_evidence, indent=2, sort_keys=True
                )
                + "\n"
            )
    update_catalog = os.environ.get("CKPT_INCREMENTAL_UPDATE_CATALOG_DIR")
    update_catalog_dir = Path(update_catalog) if update_catalog else None
    communication_profile_dir = os.environ.get("CKPT_COMMUNICATION_PROFILE_DIR")
    communication_profile = CommunicationProfile(
        enabled=communication_profile_dir is not None,
        jsonl_path=(
            Path(communication_profile_dir) / f"rank_{rank:05d}.jsonl"
            if communication_profile_dir is not None
            else None
        ),
    )
    if mode == "full_remote":
        remote = ServiceRemoteCheckpointSession(
            service_endpoint=str(endpoints[peer // ranks_per_node]),
            manifest_root=root / "manifests" / "full-remote",
            **common,
        )
        session: Any = IncrementalCheckpointSession(
            remote,
            optimizer.optimizer,
            parameter_ids=parameter_ids,
            binding=binding,
            max_inflight=max_inflight,
            update_catalog_dir=update_catalog_dir,
            run_id=run_id,
            gradient_dtype=(torch.bfloat16 if semantic_plan is not None else None),
        )
    elif mode == "peer_delta":
        local_full = ServiceRemoteCheckpointSession(
            service_endpoint=str(endpoints[rank // ranks_per_node]),
            target_rank=rank,
            manifest_root=root / "manifests" / "local-full",
            **common,
        )
        remote_delta = ServiceRemoteCheckpointSession(
            service_endpoint=str(endpoints[peer // ranks_per_node]),
            manifest_root=root / "manifests" / "remote-delta",
            **common,
        )
        local_transport = os.environ.get(
            "CKPT_INCREMENTAL_LOCAL_TRANSPORT", "byte_stream"
        )
        if local_transport not in {"byte_stream", "shared_tensor"}:
            raise ValueError("unsupported incremental local transport")
        if sealed and local_transport != "shared_tensor":
            raise ValueError("sealed peer recovery requires shared-tensor transport")
        local_update = None
        if local_transport == "shared_tensor":
            if not materialized:
                raise ValueError("shared-tensor transport requires materialization")
            from .shared_tensor import SharedTensorCheckpointSession

            socket_path = os.environ.get("CKPT_SHARED_TENSOR_SOCKET")
            if not socket_path:
                raise ValueError("CKPT_SHARED_TENSOR_SOCKET is required")
            local_update = SharedTensorCheckpointSession(
                rank=rank,
                group_ranks=group,
                target_rank=rank,
                socket_path=socket_path,
                manifest_root=root / "manifests" / "local-full",
                max_shared_slots=max_inflight,
                timeout_seconds=float(
                    os.environ.get("CKPT_REPLICA_TIMEOUT_SECONDS", "60")
                ),
            )
        session = PeerIncrementalCheckpointSession(
            local_full,
            remote_delta,
            optimizer.optimizer,
            parameter_ids=parameter_ids,
            binding=binding,
            peer_rank=peer,
            max_inflight=max_inflight,
            run_id=run_id,
            local_update_session=local_update,
            update_catalog_dir=update_catalog_dir,
            gradient_dtype=(torch.bfloat16 if semantic_plan is not None else None),
        )
    else:
        raise ValueError("CKPT_INCREMENTAL_MODE must be full_remote or peer_delta")

    return MerakIncrementalCheckpointController(
        module,
        optimizer,
        scheduler,
        mode=mode,
        session=session,
        binding=binding,
        peer_binding=peer_binding,
        parameter_ids=parameter_ids,
        epoch_provider=epoch_provider,
        runtime_capture=lambda epoch: capture_runtime_state(epoch, sampler),
        runtime_restore=lambda state: restore_runtime_state(state, sampler),
        metrics_path=root / "metrics" / f"rank_{rank:05d}.json",
        materialized=materialized,
        sealed=sealed,
        materialized_restore_source=os.environ.get(
            "CKPT_MATERIALIZED_RESTORE_SOURCE", "peer"
        ),
        run_id=run_id,
        global_commit_root=root / "global",
        max_lag_steps=int(os.environ.get("CKPT_INCREMENTAL_MAX_LAG_STEPS", "1")),
        semantic_binding=semantic_binding,
        semantic_plan=semantic_plan,
        semantic_detector_evidence=semantic_detector_evidence,
        communication_profile=communication_profile,
        global_commit_protocol=os.environ.get(
            "CKPT_GLOBAL_COMMIT_COORDINATOR", "collective-v1"
        ),
    )
