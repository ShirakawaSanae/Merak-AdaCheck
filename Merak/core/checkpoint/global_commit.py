"""Globally visible epoch control for incremental checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist

from .manifest import CheckpointManifest
from .adacheck_semantic import semantic_binding_from_rank_evidence


def manifest_evidence(manifest: CheckpointManifest) -> dict[str, Any]:
    if not manifest.committed:
        raise ValueError("global control requires committed rank manifests")
    return {
        "manifest_id": manifest.manifest_id,
        "epoch": manifest.epoch,
        "source_rank": manifest.source_rank,
        "target_rank": manifest.target_rank,
        "group_ranks": list(manifest.group_ranks),
        "payload_checksum": manifest.payload_checksum,
    }


def create_global_epoch(
    *,
    run_id: str,
    epoch: int,
    world_size: int,
    ranks: list[Mapping[str, Any]],
    publication_protocol: str | None = None,
) -> dict[str, Any]:
    if not run_id or epoch < 0 or world_size < 1 or len(ranks) != world_size:
        raise ValueError("global epoch has invalid identity or rank count")
    ordered = sorted((dict(item) for item in ranks), key=lambda item: int(item["rank"]))
    if [int(item["rank"]) for item in ordered] != list(range(world_size)):
        raise ValueError("global epoch does not contain every rank exactly once")
    modes = {item.get("mode") for item in ordered}
    if len(modes) != 1 or next(iter(modes)) not in {"full_remote", "peer_delta"}:
        raise ValueError("global rank evidence has inconsistent checkpoint modes")
    mode = str(next(iter(modes)))
    peers: dict[int, int] = {}
    groups: dict[int, tuple[int, int]] = {}
    for item in ordered:
        rank = int(item["rank"])
        paths = item.get("paths")
        required_paths = (
            {"remote_full"}
            if mode == "full_remote"
            else {"local_full", "remote_delta"}
        )
        if (
            int(item.get("epoch", -1)) != epoch
            or not isinstance(paths, Mapping)
            or set(paths) != required_paths
        ):
            raise ValueError("global rank evidence has an invalid epoch or empty paths")
        for evidence in paths.values():
            if (
                int(evidence.get("epoch", -1)) != epoch
                or int(evidence.get("source_rank", -1)) != rank
                or not evidence.get("manifest_id")
                or not evidence.get("payload_checksum")
            ):
                raise ValueError("global rank path evidence is inconsistent")
        peer_path = paths["remote_full" if mode == "full_remote" else "remote_delta"]
        peer = int(peer_path.get("target_rank", -1))
        group = tuple(int(value) for value in peer_path.get("group_ranks", ()))
        if (
            peer == rank
            or not 0 <= peer < world_size
            or len(group) != 2
            or tuple(sorted(group)) != group
            or set(group) != {rank, peer}
        ):
            raise ValueError("global rank evidence has an invalid peer placement")
        if mode == "peer_delta":
            local = paths["local_full"]
            if (
                int(local.get("target_rank", -1)) != rank
                or tuple(int(value) for value in local.get("group_ranks", ())) != group
            ):
                raise ValueError("global local-full evidence has an invalid placement")
        peers[rank] = peer
        groups[rank] = group  # type: ignore[assignment]
    if any(
        peers.get(peer) != rank or groups.get(peer) != groups[rank]
        for rank, peer in peers.items()
    ):
        raise ValueError("global rank evidence does not contain reciprocal peer pairs")
    if publication_protocol == "pin-before-publish-v1" and mode == "peer_delta":
        tokens = {
            int(item["rank"]): item.get("peer_equivalence_checksum")
            for item in ordered
        }
        if any(
            not isinstance(token, str)
            or len(token) != 64
            or any(character not in "0123456789abcdef" for character in token)
            for token in tokens.values()
        ):
            raise ValueError("sealed peer recovery requires per-epoch equivalence evidence")
        if any(tokens[rank] != tokens[peer] for rank, peer in peers.items()):
            raise ValueError("sealed checkpoint peers are not equivalent at this epoch")
    body = {
        "schema_version": 2,
        "kind": "merak_global_incremental_commit",
        "run_id": run_id,
        "epoch": epoch,
        "world_size": world_size,
        "mode": mode,
        "ranks": ordered,
    }
    semantic = semantic_binding_from_rank_evidence(ordered)
    if semantic is not None:
        body["adacheck_semantic"] = semantic
    encoders = [item.get("adacheck_semantic_encoder") for item in ordered]
    if any(value is not None for value in encoders):
        if any(not isinstance(value, Mapping) for value in encoders):
            raise ValueError("global epoch has partial SemanticFixed evidence")
        common = [
            {key: item for key, item in value.items() if key != "rank_action_count"}
            for value in encoders
        ]
        if any(value != common[0] for value in common[1:]):
            raise ValueError("global epoch has inconsistent SemanticFixed evidence")
        if sum(int(value["rank_action_count"]) for value in encoders) != int(
            common[0]["global_action_count"]
        ):
            raise ValueError("global SemanticFixed action counts are incomplete")
        body["adacheck_semantic_encoder"] = common[0]
    if publication_protocol is not None:
        if publication_protocol != "pin-before-publish-v1":
            raise ValueError("unsupported global publication protocol")
        body["publication_protocol"] = publication_protocol
    checksum = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**body, "control_checksum": checksum}


def validate_global_epoch(
    value: Mapping[str, Any], *, run_id: str, epoch: int, world_size: int
) -> dict[str, Any]:
    control = dict(value)
    checksum = control.pop("control_checksum", None)
    actual = hashlib.sha256(
        json.dumps(control, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if checksum != actual:
        raise ValueError("global epoch control checksum mismatch")
    if (
        control.get("schema_version") != 2
        or control.get("kind") != "merak_global_incremental_commit"
        or control.get("run_id") != run_id
        or int(control.get("epoch", -1)) != epoch
        or int(control.get("world_size", -1)) != world_size
    ):
        raise ValueError("global epoch control identity mismatch")
    recreated = create_global_epoch(
        run_id=run_id,
        epoch=epoch,
        world_size=world_size,
        ranks=control["ranks"],
        publication_protocol=control.get("publication_protocol"),
    )
    if control.get("adacheck_semantic") != recreated.get("adacheck_semantic"):
        raise ValueError("global epoch AdaCheck semantic binding mismatch")
    if control.get("adacheck_semantic_encoder") != recreated.get(
        "adacheck_semantic_encoder"
    ):
        raise ValueError("global epoch SemanticFixed evidence mismatch")
    return recreated


class GlobalCommitRepository:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def publish(self, control: Mapping[str, Any]) -> None:
        epoch = int(control["epoch"])
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"epoch_{epoch:08d}.json"
        temporary = self.root / f".{destination.name}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)

    def load(self, *, run_id: str, epoch: int, world_size: int) -> dict[str, Any]:
        path = self.root / f"epoch_{epoch:08d}.json"
        return validate_global_epoch(
            json.loads(path.read_text()),
            run_id=run_id,
            epoch=epoch,
            world_size=world_size,
        )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class FilesystemGlobalCommitCoordinator:
    """Coordinate sealed global epochs off the training/NCCL critical path."""

    protocol = "filesystem-background-v1"

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        run_id: str,
        root: Path | str,
        initial_epoch: int = -1,
        before_publish: Callable[[Mapping[str, Any]], None] | None = None,
        after_publish: Callable[[Mapping[str, Any]], None] | None = None,
        timeout_seconds: float = 600,
        poll_interval_seconds: float = 0.02,
    ) -> None:
        if (before_publish is None) != (after_publish is None):
            raise ValueError("global sealing requires both publication callbacks")
        if not 0 <= rank < world_size:
            raise ValueError("filesystem coordinator rank is outside the world")
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("filesystem coordinator timeouts must be positive")
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.repository = GlobalCommitRepository(root)
        self.before_publish = before_publish
        self.after_publish = after_publish
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.scheduled_through = initial_epoch
        self.published_through = initial_epoch
        self._worker_through = initial_epoch
        self._target_epoch = initial_epoch
        self._closed = False
        self._failure: BaseException | None = None
        self._condition = threading.Condition()
        self._coordination_root = self.repository.root / ".filesystem-background-v1"
        self._worker = threading.Thread(
            target=self._coordinate_loop,
            name=f"checkpoint-global-commit-rank-{rank}",
            daemon=True,
        )
        self._worker.start()

    def _epoch_root(self, epoch: int) -> Path:
        return self._coordination_root / f"epoch_{epoch:08d}"

    def _rank_path(self, epoch: int, phase: str, rank: int | None = None) -> Path:
        selected = self.rank if rank is None else rank
        return self._epoch_root(epoch) / phase / f"rank_{selected:05d}.json"

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("filesystem global checkpoint commit failed") from self._failure

    def _remote_failure(self, epoch: int) -> RuntimeError | None:
        paths = sorted((self._epoch_root(epoch) / "errors").glob("rank_*.json"))
        if not paths:
            return None
        value = json.loads(paths[0].read_text())
        return RuntimeError(
            f"rank {value.get('rank')} {value.get('phase')} failed: {value.get('error')}"
        )

    def _wait_for(self, predicate: Callable[[], bool], *, epoch: int, phase: str) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        while not predicate():
            remote = self._remote_failure(epoch)
            if remote is not None:
                raise remote
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"filesystem global commit epoch {epoch} timed out in {phase}"
                )
            time.sleep(self.poll_interval_seconds)

    def _all_rank_paths_exist(self, epoch: int, phase: str) -> bool:
        return all(
            self._rank_path(epoch, phase, rank).is_file()
            for rank in range(self.world_size)
        )

    def _load_control(self, path: Path, epoch: int) -> dict[str, Any]:
        return validate_global_epoch(
            json.loads(path.read_text()),
            run_id=self.run_id,
            epoch=epoch,
            world_size=self.world_size,
        )

    def _coordinate_epoch(self, epoch: int) -> None:
        epoch_root = self._epoch_root(epoch)
        proposal = epoch_root / "proposal.json"
        published = self.repository.root / f"epoch_{epoch:08d}.json"
        done = epoch_root / "done.json"

        self._wait_for(
            lambda: self._all_rank_paths_exist(epoch, "evidence"),
            epoch=epoch,
            phase="evidence",
        )
        if self.rank == 0 and not proposal.exists():
            ranks = [
                json.loads(self._rank_path(epoch, "evidence", rank).read_text())
                for rank in range(self.world_size)
            ]
            control = create_global_epoch(
                run_id=self.run_id,
                epoch=epoch,
                world_size=self.world_size,
                ranks=ranks,
                publication_protocol=(
                    "pin-before-publish-v1"
                    if self.before_publish is not None
                    else None
                ),
            )
            _atomic_write_json(proposal, control)
        self._wait_for(
            proposal.is_file, epoch=epoch, phase="proposal"
        )
        control = self._load_control(proposal, epoch)

        if self.before_publish is not None:
            self.before_publish(control)
        _atomic_write_json(
            self._rank_path(epoch, "pinned"),
            {"rank": self.rank, "control_checksum": control["control_checksum"]},
        )

        if self.rank == 0:
            self._wait_for(
                lambda: self._all_rank_paths_exist(epoch, "pinned"),
                epoch=epoch,
                phase="pin",
            )
            self.repository.publish(control)
        self._wait_for(published.is_file, epoch=epoch, phase="publication")
        published_control = self._load_control(published, epoch)
        if published_control != control:
            raise ValueError("published global checkpoint differs from its proposal")

        if self.after_publish is not None:
            self.after_publish(control)
        _atomic_write_json(
            self._rank_path(epoch, "sealed"),
            {"rank": self.rank, "control_checksum": control["control_checksum"]},
        )

        if self.rank == 0:
            self._wait_for(
                lambda: self._all_rank_paths_exist(epoch, "sealed"),
                epoch=epoch,
                phase="seal",
            )
            _atomic_write_json(
                done,
                {
                    "epoch": epoch,
                    "control_checksum": control["control_checksum"],
                },
            )
        self._wait_for(done.is_file, epoch=epoch, phase="completion")
        completion = json.loads(done.read_text())
        if (
            int(completion.get("epoch", -1)) != epoch
            or completion.get("control_checksum") != control["control_checksum"]
        ):
            raise ValueError("filesystem global checkpoint completion is invalid")

    def _record_failure(self, epoch: int, phase: str, error: BaseException) -> None:
        try:
            _atomic_write_json(
                self._rank_path(epoch, "errors"),
                {
                    "rank": self.rank,
                    "epoch": epoch,
                    "phase": phase,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        except Exception:
            pass

    def _coordinate_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._failure is not None
                    or self._worker_through < self._target_epoch
                    or self._closed
                )
                if self._failure is not None:
                    return
                if self._worker_through >= self._target_epoch:
                    if self._closed:
                        return
                    continue
                epoch = self._worker_through + 1
            try:
                self._coordinate_epoch(epoch)
            except BaseException as error:
                self._record_failure(epoch, "coordinate", error)
                with self._condition:
                    self._failure = error
                    self._condition.notify_all()
                return
            with self._condition:
                self._worker_through = epoch
                self.published_through = epoch
                self._condition.notify_all()

    def submit_epoch(self, epoch: int, local_evidence: Mapping[str, Any]) -> None:
        if epoch != self.scheduled_through + 1:
            raise ValueError("global commit epochs must be scheduled contiguously")
        self._raise_if_failed()
        _atomic_write_json(self._rank_path(epoch, "evidence"), dict(local_evidence))
        self.scheduled_through = epoch
        with self._condition:
            self._target_epoch = epoch
            self._condition.notify_all()

    def advance(
        self,
        local_latest: int,
        evidence: Callable[[int], Mapping[str, Any]],
    ) -> int:
        self._raise_if_failed()
        for epoch in range(self.scheduled_through + 1, local_latest + 1):
            self.submit_epoch(epoch, evidence(epoch))
        with self._condition:
            return self.published_through

    def wait_through(self, epoch: int) -> int:
        deadline = time.monotonic() + self.timeout_seconds
        with self._condition:
            while self.published_through < epoch and self._failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"filesystem global commit did not complete epoch {epoch}"
                    )
                self._condition.wait(min(remaining, self.poll_interval_seconds))
        self._raise_if_failed()
        return self.published_through

    def close(self) -> int:
        if self._closed:
            self._raise_if_failed()
            return self.published_through
        target = self.scheduled_through
        self.wait_through(target)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join()
        self._raise_if_failed()
        return self.published_through


class GlobalCommitCoordinator:
    """Collect rank evidence, pin it, and publish complete global epochs."""

    protocol = "collective-v1"

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        run_id: str,
        root: Path | str,
        initial_epoch: int = -1,
        before_publish: Callable[[Mapping[str, Any]], None] | None = None,
        after_publish: Callable[[Mapping[str, Any]], None] | None = None,
    ):
        if (before_publish is None) != (after_publish is None):
            raise ValueError("global sealing requires both publication callbacks")
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.repository = GlobalCommitRepository(root)
        self.scheduled_through = initial_epoch
        self.published_through = initial_epoch
        self.before_publish = before_publish
        self.after_publish = after_publish

    def submit_epoch(self, epoch: int, local_evidence: Mapping[str, Any]) -> None:
        if epoch != self.scheduled_through + 1:
            raise ValueError("global commit epochs must be scheduled contiguously")
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, dict(local_evidence))
        error = None
        control: dict[str, Any] | None = None
        if self.rank == 0:
            try:
                control = create_global_epoch(
                    run_id=self.run_id,
                    epoch=epoch,
                    world_size=self.world_size,
                    ranks=gathered,
                    publication_protocol=(
                        "pin-before-publish-v1"
                        if self.before_publish is not None
                        else None
                    ),
                )
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
        self._raise_collective_error(error)
        controls: list[Any] = [control]
        dist.broadcast_object_list(
            controls, src=0, device=self._collective_device()
        )
        control = controls[0]
        if self.before_publish is not None:
            self._run_collective_callback("pin", self.before_publish, control)
        error = None
        if self.rank == 0:
            try:
                self.repository.publish(control)
                self.published_through = epoch
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
        self._raise_collective_error(error)
        if self.after_publish is not None:
            self._run_collective_callback("seal", self.after_publish, control)
        self.scheduled_through = epoch

    def advance(
        self,
        local_latest: int,
        evidence: Callable[[int], Mapping[str, Any]],
    ) -> int:
        ready = torch.tensor(local_latest, device=self._collective_device())
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        for epoch in range(self.scheduled_through + 1, int(ready.item()) + 1):
            self.submit_epoch(epoch, evidence(epoch))
        return self._sync_published(wait=False)

    @staticmethod
    def _collective_device() -> torch.device:
        return (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )

    def close(self) -> int:
        return self._sync_published(wait=True)

    def _sync_published(self, *, wait: bool) -> int:
        error = None
        self._raise_collective_error(error)
        published = torch.tensor(
            self.published_through, device=self._collective_device()
        )
        dist.broadcast(published, src=0)
        self.published_through = int(published.item())
        return self.published_through

    def _raise_collective_error(self, error: str | None) -> None:
        message = [error if self.rank == 0 else None]
        dist.broadcast_object_list(
            message, src=0, device=self._collective_device()
        )
        if message[0] is not None:
            raise RuntimeError(f"global checkpoint commit failed: {message[0]}")

    def _run_collective_callback(
        self,
        phase: str,
        callback: Callable[[Mapping[str, Any]], None],
        control: Mapping[str, Any],
    ) -> None:
        error = None
        try:
            callback(control)
        except Exception as caught:
            error = f"rank {self.rank} {type(caught).__name__}: {caught}"
        errors: list[Any] = [None] * self.world_size
        dist.all_gather_object(errors, error)
        failure = next((item for item in errors if item is not None), None)
        if failure is not None:
            raise RuntimeError(f"global checkpoint {phase} failed: {failure}")


def rank_evidence(control: Mapping[str, Any], rank: int) -> Mapping[str, Any]:
    try:
        return next(item for item in control["ranks"] if int(item["rank"]) == rank)
    except StopIteration as error:
        raise ValueError(f"global epoch has no evidence for rank {rank}") from error
