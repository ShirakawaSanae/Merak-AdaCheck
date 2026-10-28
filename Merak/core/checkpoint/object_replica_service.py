"""Replica-service storage with immutable per-epoch ObjectRecovery seals."""

from __future__ import annotations

from typing import Any, Mapping

from .matched import CONTROL_SOURCE_RANK
from .replica_service import ServiceReplicaStore


class ObjectReplicaServiceStore(ServiceReplicaStore):
    """Keep every sealed object-update epoch readable by its own control hash."""

    def __init__(
        self,
        max_bytes: int,
        retained_epochs_per_source: int = 2,
        *,
        sealed_incremental: bool = True,
    ) -> None:
        if not sealed_incremental:
            raise ValueError("object replica storage requires sealed incremental mode")
        super().__init__(
            max_bytes,
            retained_epochs_per_source,
            materialize_incremental=False,
            sealed_incremental=False,
        )
        # ObjectRecovery supplies its own seal history and does not use the
        # parent's rank-local AdamW materializer.
        self.sealed_incremental = True
        self._object_sealed: dict[tuple[str, int, int], str] = {}

    def _protected_record_keys(self) -> set[tuple[int, int]]:
        protected = super()._protected_record_keys()
        protected.update(
            (source_rank, epoch)
            for (kind, source_rank, epoch) in self._object_sealed
            if kind == "record"
        )
        return protected

    def _authorize(
        self, kind: str, source_rank: int, epoch: int, control_checksum: str | None
    ) -> None:
        if (
            control_checksum is None
            or self._object_sealed.get((kind, source_rank, epoch))
            != control_checksum
        ):
            raise ValueError("object checkpoint epoch is not sealed by this control")

    def pin(
        self,
        *,
        kind: str,
        source_rank: int,
        epoch: int,
        control_checksum: str,
        expected_manifest: Mapping[str, Any],
    ) -> None:
        if kind != "record":
            raise ValueError("object replica storage pins serialized records only")
        if source_rank == CONTROL_SOURCE_RANK:
            with self._lock:
                try:
                    checksum = self._records[(source_rank, epoch)].manifest.payload_checksum
                except KeyError as error:
                    raise KeyError("control record not found") from error
                if control_checksum != checksum:
                    raise ValueError("control pin must use its payload checksum")
        super().pin(
            kind=kind,
            source_rank=source_rank,
            epoch=epoch,
            control_checksum=control_checksum,
            expected_manifest=expected_manifest,
        )

    def seal(
        self, *, kind: str, source_rank: int, epoch: int, control_checksum: str
    ) -> None:
        if kind != "record":
            raise ValueError("object replica storage seals serialized records only")
        pin = (kind, source_rank, epoch)
        with self._lock:
            if source_rank == CONTROL_SOURCE_RANK:
                try:
                    checksum = self._records[(source_rank, epoch)].manifest.payload_checksum
                except KeyError as error:
                    raise KeyError("control record not found") from error
                if control_checksum != checksum:
                    raise ValueError("control seal must use its payload checksum")
            elif self._object_sealed.get(
                (kind, CONTROL_SOURCE_RANK, epoch)
            ) != control_checksum:
                raise ValueError("data seal requires its sealed control record")
            if self._pins.get(pin) != control_checksum:
                raise ValueError("seal has no matching pin")
            previous = self._object_sealed.get(pin)
            if previous is not None and previous != control_checksum:
                raise ValueError("object checkpoint epoch is already sealed")
            self._object_sealed[pin] = control_checksum
            self._trim_records(source_rank)

    def get_committed(
        self,
        source_rank: int,
        epoch: int,
        *,
        control_checksum: str | None = None,
        expected_manifest: Mapping[str, Any] | None = None,
    ):
        if expected_manifest is None:
            raise ValueError("object checkpoint GET requires manifest evidence")
        return super().get_committed(
            source_rank,
            epoch,
            control_checksum=control_checksum,
            expected_manifest=expected_manifest,
        )

    def stats(self) -> dict[str, Any]:
        value = super().stats()
        with self._lock:
            value["object_sealed_epochs"] = [
                [kind, source_rank, epoch]
                for kind, source_rank, epoch in sorted(self._object_sealed)
            ]
        return value
