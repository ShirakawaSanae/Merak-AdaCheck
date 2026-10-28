"""Background runtime adapter for the independent SemanticFixed object path."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import time
from typing import Any, Literal

import torch
import torch.distributed as dist

from ..amp.amp_optimizer import HalfPrecisionOptimizer
from .async_checkpoint import AsyncServiceRemoteCheckpointSession
from .incremental_checkpoint import RecoveryBinding, capture_optimizer_update
from .object_optimizer_update import (
    ObjectOptimizerUpdate,
    externalize_sparse_owned_snapshot_optimizer_update,
)
from .object_semantic_dp_invariant import (
    gather_dp_invariant_gradient_encoding,
    staged_object_ids_by_rank,
    verifier_object_ids_by_rank,
)
from .object_semantic_plan import ActiveGradientPlan, ConfirmedGradientEncoding
from .object_update_distributed import commit_owned_snapshot_object_update_epoch


SemanticArm = Literal["adacheck_fixed", "host_global_fixed", "m1"]
SEMANTIC_ARMS: tuple[SemanticArm, ...] = (
    "adacheck_fixed",
    "host_global_fixed",
    "m1",
)


def create_world_gloo_control_group() -> object:
    """Create the dedicated, world-size Gloo metadata group on every rank."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("SemanticFixed runtime requires torch.distributed")
    return dist.new_group(ranks=list(range(dist.get_world_size())), backend="gloo")


def _require_world_gloo(group: object) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("SemanticFixed runtime requires an initialized process group")
    if dist.get_backend(group) != "gloo":
        raise ValueError("SemanticFixed runtime requires a Gloo control group")
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    if rank != dist.get_rank() or world_size != dist.get_world_size():
        raise ValueError("SemanticFixed runtime requires a world-size Gloo control group")
    return rank, world_size


def _cpu_only(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type == "cpu"
    if isinstance(value, Mapping):
        return all(_cpu_only(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_cpu_only(item) for item in value)
    return True


class _ObjectCommitSession:
    """AsyncService adapter whose save method runs on the FIFO worker."""

    repository = None

    def __init__(self, controller: "SemanticFixedRuntimeController") -> None:
        self.controller = controller
        self.rank = controller.binding.rank

    def save(
        self,
        epoch: int,
        record: Mapping[str, Any],
        *,
        state_is_snapshot: bool,
        object_checksums: bool,
    ) -> Any:
        if not state_is_snapshot or not object_checksums:
            raise ValueError("SemanticFixed worker requires checksummed CPU snapshots")
        return self.controller._publish_cpu_snapshot(epoch, record)


class SemanticFixedRuntimeController:
    """Capture raw BF16 updates and commit them from the async FIFO worker.

    The callback only creates an update record and enqueues it.  The async
    session copies it to CPU, waits for its CUDA event when needed, then calls
    this adapter's ``save`` method on its worker.  ``poll`` only consumes
    completed results; it never invokes collectives or network I/O.
    """

    def __init__(
        self,
        optimizer: HalfPrecisionOptimizer,
        *,
        arm: SemanticArm,
        plan: ActiveGradientPlan,
        binding: RecoveryBinding,
        parameter_ids: Sequence[Sequence[str]],
        base_id: str,
        epoch_provider: Callable[[], int],
        runtime_capture: Callable[[int], Mapping[str, Any]],
        scheduler: Any,
        control_group: object,
        clients: Mapping[int, Any],
        run_id: str,
        locator_root: str | Path,
        max_inflight: int = 2,
        group_validator: Callable[[object], tuple[int, int]] = _require_world_gloo,
        checksum: Callable[
            ..., ConfirmedGradientEncoding
        ] = gather_dp_invariant_gradient_encoding,
        externalize: Callable[
            ..., ObjectOptimizerUpdate
        ] = externalize_sparse_owned_snapshot_optimizer_update,
        commit: Callable[..., Any] = commit_owned_snapshot_object_update_epoch,
        profile: Callable[..., None] | None = None,
    ) -> None:
        if arm not in SEMANTIC_ARMS or plan.arm != arm:
            raise ValueError("SemanticFixed runtime arm must match its exact three-arm plan")
        if not isinstance(optimizer, HalfPrecisionOptimizer):
            raise TypeError("SemanticFixed runtime requires HalfPrecisionOptimizer")
        if type(optimizer.optimizer) is not torch.optim.AdamW:
            raise TypeError("SemanticFixed runtime requires torch.optim.AdamW")
        for group in optimizer.optimizer.param_groups:
            if group.get("foreach") is not False or group.get("fused") not in (None, False):
                raise ValueError(
                    "SemanticFixed runtime requires non-fused AdamW foreach=False"
                )
            if (
                group.get("amsgrad", False)
                or group.get("capturable", False)
                or group.get("differentiable", False)
            ):
                raise ValueError("SemanticFixed runtime rejects unsupported AdamW semantics")
        if binding.world_size != len(plan.rank_bindings):
            raise ValueError("SemanticFixed runtime binding and plan world sizes differ")
        if set(clients) != {0, 1} or not run_id:
            raise ValueError("SemanticFixed runtime requires two clients and a run ID")
        self.optimizer = optimizer
        self.arm = arm
        self.plan = plan
        self.binding = binding
        self.parameter_ids = tuple(tuple(group) for group in parameter_ids)
        self.base_id = base_id
        self.epoch_provider = epoch_provider
        self.runtime_capture = runtime_capture
        self.scheduler = scheduler
        self.control_group = control_group
        self.clients = clients
        self.run_id = run_id
        self.locator_root = Path(locator_root)
        self.max_inflight = max_inflight
        self.group_validator = group_validator
        self.checksum = checksum
        self.externalize = externalize
        self.commit = commit
        self.profile = profile
        staged = staged_object_ids_by_rank(plan)
        verifiers = verifier_object_ids_by_rank(plan)
        self._staged_object_ids = frozenset(staged[binding.rank])
        self._verifier_object_ids = frozenset(verifiers[binding.rank])
        rank_objects = {
            item.fqn: item for item in plan.objects if binding.rank in item.holders
        }
        if set(rank_objects) != {
            parameter_id for group in self.parameter_ids for parameter_id in group
        }:
            raise ValueError("SemanticFixed parameter IDs differ from its object plan")
        self._rank_objects = rank_objects
        self._staged_fqns = frozenset(
            item.fqn
            for item in rank_objects.values()
            if item.object_id in self._staged_object_ids
        )
        self.base_epoch: int | None = None
        self.captured_epochs: list[int] = []
        self.committed_epochs: list[int] = []
        self.completed_results: list[Any] = []
        self.boundary_lag: list[int] = []
        self._published_epochs: list[int] = []
        self._publish_timings: dict[int, dict[str, float | None]] = {}
        self._rank: int | None = None
        self._async: AsyncServiceRemoteCheckpointSession | None = None

    def record_profile_interval(
        self, name: str, start_ns: int, end_ns: int, **fields: Any
    ) -> None:
        if self.profile is not None:
            self.profile(name, start_ns, end_ns, **fields)

    def start(self, epoch: int) -> None:
        if self.base_epoch is not None:
            raise RuntimeError("SemanticFixed runtime is already started")
        rank, world_size = self.group_validator(self.control_group)
        if rank != self.binding.rank or world_size != self.binding.world_size:
            raise ValueError("SemanticFixed runtime binding disagrees with the control group")
        if self.optimizer.global_step != epoch:
            raise ValueError("SemanticFixed runtime optimizer and base epochs disagree")
        self.base_epoch, self._rank = int(epoch), rank
        preallocation_started = time.monotonic_ns()
        self._async = AsyncServiceRemoteCheckpointSession(
            _ObjectCommitSession(self),
            max_inflight=self.max_inflight,
            initial_slot_state_factory=self._initial_slot_state,
        )
        self.record_profile_interval(
            "checkpoint_object_semantic_snapshot_preallocation",
            preallocation_started,
            time.monotonic_ns(),
            arm=self.arm,
            max_inflight=self.max_inflight,
            allocated_bytes=self.max_inflight
            * sum(
                parameter.numel() * parameter.element_size()
                for ids, group in zip(self.parameter_ids, self.optimizer.fp16_groups)
                for parameter_id, parameter in zip(ids, group)
                if parameter_id in self._staged_fqns
            ),
            confirmation_mode="post_allreduce_dp_invariant",
        )
        self.optimizer.set_pre_optimizer_step_callback(
            self._capture_raw_update, capture_raw_gradients=True
        )

    def _initial_slot_state(self) -> dict[str, Any]:
        return {
            "gradients": [
                [
                    (
                        torch.empty_like(
                            parameter,
                            device="cpu",
                            pin_memory=parameter.device.type == "cuda",
                        )
                        if parameter_id in self._staged_fqns
                        else None
                    )
                    for parameter_id, parameter in zip(ids, group)
                ]
                for ids, group in zip(self.parameter_ids, self.optimizer.fp16_groups)
            ]
        }

    def _capture_raw_update(
        self,
        optimizer_epoch: int,
        inner_optimizer: torch.optim.Optimizer,
        gradient_multiplier: float,
    ) -> None:
        if (
            self.base_epoch is None
            or self._async is None
            or inner_optimizer is not self.optimizer.optimizer
        ):
            raise RuntimeError("SemanticFixed runtime callback is not active")
        epoch = int(self.epoch_provider())
        expected = self.base_epoch + len(self.captured_epochs) + 1
        if optimizer_epoch != epoch or epoch != expected:
            raise ValueError("SemanticFixed runtime callback epoch is not contiguous")
        runtime_state = self.runtime_capture(epoch)
        if not isinstance(runtime_state, Mapping):
            raise ValueError("SemanticFixed runtime capture must return a mapping")
        record = capture_optimizer_update(
            inner_optimizer,
            parameter_ids=self.parameter_ids,
            binding=self.binding,
            base_id=self.base_id,
            base_epoch=self.base_epoch,
            epoch=epoch,
            metadata={
                "trainer_global_step": epoch,
                "scheduler_state_before_step": self.scheduler.state_dict(),
            },
            runtime_state=runtime_state,
            snapshot_gradients=False,
            gradient_multiplier=gradient_multiplier,
        )
        raw_gradients = []
        if len(self.optimizer.fp16_groups) != len(inner_optimizer.param_groups):
            raise ValueError("SemanticFixed raw and master gradient groups differ")
        for ids, raw_group, master_group in zip(
            self.parameter_ids, self.optimizer.fp16_groups, inner_optimizer.param_groups
        ):
            if len(raw_group) != len(master_group["params"]):
                raise ValueError("SemanticFixed raw and master gradient counts differ")
            current = []
            for parameter_id, raw_parameter, master_parameter in zip(
                ids, raw_group, master_group["params"]
            ):
                raw, master = raw_parameter.grad, master_parameter.grad
                if (raw is None) != (master is None):
                    raise ValueError("SemanticFixed raw and master gradients disagree")
                if raw is not None and (
                    raw.dtype != torch.bfloat16 or raw.shape != master.shape
                ):
                    raise ValueError("SemanticFixed requires matching raw BF16 gradients")
                if parameter_id in self._staged_fqns and raw is None:
                    raise ValueError("SemanticFixed staged gradient is absent")
                current.append(
                    raw.detach()
                    if raw is not None and parameter_id in self._staged_fqns
                    else None
                )
            raw_gradients.append(current)
        record["gradients"] = raw_gradients
        record["gradient_dtype"] = str(torch.bfloat16)
        self._async.enqueue(epoch, record)
        self.captured_epochs.append(epoch)

    def _local_tensors(self, record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        assert self._rank is not None
        by_fqn = {
            parameter_id: gradient
            for ids, gradients in zip(
                record.get("parameter_ids", ()), record.get("gradients", ())
            )
            for parameter_id, gradient in zip(ids, gradients)
        }
        expected = {
            item.fqn: item.object_id
            for item in self.plan.objects
            if item.object_id in self._verifier_object_ids
        }
        non_null = {name for name, value in by_fqn.items() if value is not None}
        if non_null != self._staged_fqns or any(
            not isinstance(by_fqn[name], torch.Tensor) for name in expected
        ):
            raise ValueError("CPU snapshot gradients do not match SemanticFixed staging")
        return {object_id: by_fqn[fqn] for fqn, object_id in expected.items()}

    def _locator_path(self, epoch: int) -> Path:
        return self.locator_root / f"epoch-{epoch:08d}.json"

    def _publish_cpu_snapshot(self, epoch: int, record: Mapping[str, Any]) -> Any:
        assert self.base_epoch is not None and self._rank is not None
        if int(record.get("epoch", -1)) != epoch or not _cpu_only(record):
            raise ValueError("SemanticFixed worker received an invalid CPU snapshot")
        expected = self.base_epoch + len(self._published_epochs) + 1
        if epoch != expected:
            raise ValueError("SemanticFixed CPU snapshots must publish contiguously")
        local_tensors = self._local_tensors(record)
        checksum_started = time.monotonic_ns()
        encoding = self.checksum(
            self.plan,
            local_tensors,
            epoch=epoch,
            group=self.control_group,
        )
        checksum_finished = time.monotonic_ns()
        self.record_profile_interval(
            "checkpoint_object_semantic_checksum_confirmation",
            checksum_started,
            checksum_finished,
            arm=self.arm,
            epoch=epoch,
        )
        externalize_started = time.monotonic_ns()
        update = self.externalize(self.plan, encoding, self._rank, record)
        externalize_finished = time.monotonic_ns()
        self.record_profile_interval(
            "checkpoint_object_semantic_externalize",
            externalize_started,
            externalize_finished,
            arm=self.arm,
            epoch=epoch,
        )
        commit_started = time.monotonic_ns()
        result = self.commit(
            self.clients,
            run_id=self.run_id,
            plan=self.plan,
            encoding=encoding,
            update=update,
            locator_path=self._locator_path(epoch),
            group=self.control_group,
        )
        commit_finished = time.monotonic_ns()
        self.record_profile_interval(
            "checkpoint_object_semantic_distributed_commit",
            commit_started,
            commit_finished,
            arm=self.arm,
            epoch=epoch,
        )
        timing: dict[str, float | None] = {
            "checksum_confirmation_ms": (checksum_finished - checksum_started) / 1_000_000,
            "externalize_ms": (externalize_finished - externalize_started) / 1_000_000,
            "distributed_commit_ms": (commit_finished - commit_started) / 1_000_000,
        }
        result_timing = getattr(result, "timing", None)
        if isinstance(result_timing, Mapping):
            timing.update(result_timing)
        self._publish_timings[epoch] = timing
        self._published_epochs.append(epoch)
        return result

    def _accept(self, completed: Sequence[Any]) -> tuple[Any, ...]:
        assert self.base_epoch is not None
        for item in completed:
            expected = self.base_epoch + len(self.committed_epochs) + 1
            if int(item.epoch) != expected:
                raise ValueError("SemanticFixed completed epochs are not contiguous")
            self.committed_epochs.append(expected)
            self.completed_results.append(item)
        return tuple(completed)

    def poll(self, current_epoch: int | None = None) -> tuple[Any, ...]:
        """Consume completed worker results; this method performs no commit work."""
        if self._async is None:
            return ()
        completed = self._accept(self._async.poll())
        if current_epoch is not None:
            latest = max(self.committed_epochs, default=self.base_epoch or 0)
            self.boundary_lag.append(int(current_epoch) - latest)
        return completed

    def zero_grad(self, *, set_grads_to_none: bool = True) -> None:
        if set_grads_to_none is not True:
            raise ValueError("SemanticFixed runtime requires zero_grad(set_grads_to_none=True)")
        self.optimizer.zero_grad(set_grads_to_none=True)

    def finalize(self, current_epoch: int | None = None) -> dict[str, Any]:
        if self.base_epoch is None or self._async is None:
            return {}
        self.optimizer.set_pre_optimizer_step_callback(None)
        drain_started = time.monotonic_ns()
        self._accept(self._async.close())
        drain_finished = time.monotonic_ns()
        self.record_profile_interval(
            "checkpoint_object_semantic_final_drain",
            drain_started,
            drain_finished,
            arm=self.arm,
            final_epoch=current_epoch,
        )
        expected = list(
            range(self.base_epoch + 1, self.base_epoch + len(self.captured_epochs) + 1)
        )
        final_epoch = self.base_epoch if not expected else expected[-1]
        valid = (
            self.captured_epochs == expected
            and self.committed_epochs == expected
            and (current_epoch is None or int(current_epoch) == final_epoch)
        )
        maximum_boundary_lag = max(self.boundary_lag, default=0)
        strict_1c1s_valid = valid and maximum_boundary_lag <= 1
        commit_metrics = []
        for item in self.completed_results:
            value = {
                "epoch": int(item.epoch),
                "capture_ms": float(item.capture_ms),
                "publisher_queue_ms": float(item.publisher_queue_ms),
                "snapshot_ready_wait_ms": float(item.snapshot_ready_wait_ms),
                "enqueue_to_commit_ms": float(item.enqueue_to_commit_ms),
            }
            value.update(self._publish_timings.get(int(item.epoch), {}))
            transfer = getattr(item.result, "metrics", None)
            if transfer is not None:
                for name in (
                    "remote_gradient_logical_bytes",
                    "remote_data_payload_bytes",
                    "remote_control_payload_bytes",
                    "remote_wire_bytes",
                    "maximum_remote_put_ms",
                    "total_remote_payload_bytes",
                ):
                    if hasattr(transfer, name):
                        value[name] = getattr(transfer, name)
            commit_metrics.append(value)
        return {
            "arm": self.arm,
            "plan_load_metrics": {
                "confirmation_mode": "post_allreduce_dp_invariant",
                "local_staged_logical_bytes": sum(
                    item.logical_bytes
                    for item in self.plan.objects
                    if item.object_id in self._staged_object_ids
                ),
                "local_verifier_logical_bytes": sum(
                    item.logical_bytes
                    for item in self.plan.objects
                    if item.object_id in self._verifier_object_ids
                ),
                "publisher_policy": self.plan.publisher_policy,
                "all_publication_publisher_bytes": [
                    [rank, logical_bytes]
                    for rank, logical_bytes in self.plan.all_publication_publisher_bytes
                ],
                "total_all_publication_logical_bytes": (
                    self.plan.total_all_publication_logical_bytes
                ),
                "maximum_all_publication_publisher_bytes": (
                    self.plan.maximum_all_publication_publisher_bytes
                ),
                "total_remote_logical_bytes": self.plan.total_remote_logical_bytes,
                "maximum_remote_publisher_bytes": (
                    self.plan.maximum_remote_publisher_bytes
                ),
            },
            "base_epoch": self.base_epoch,
            "captured_epochs": self.captured_epochs,
            "committed_epochs": self.committed_epochs,
            "boundary_lag": self.boundary_lag,
            "maximum_boundary_lag": maximum_boundary_lag,
            "final_drain_ms": (drain_finished - drain_started) / 1_000_000,
            "commit_metrics": commit_metrics,
            "valid": valid,
            "all_epochs_committed": valid,
            "recovery_valid": valid,
            "cadence_valid": strict_1c1s_valid,
            "strict_1c1s_valid": strict_1c1s_valid,
        }
