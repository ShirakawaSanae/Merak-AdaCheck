"""CPU checkpoint snapshots and object-level integrity metadata."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch


@dataclass(frozen=True)
class CheckpointObject:
    object_id: str
    path: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    logical_bytes: int
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "path": self.path,
            "role": self.role,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "logical_bytes": self.logical_bytes,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointObject":
        return cls(
            object_id=str(value["object_id"]),
            path=str(value["path"]),
            role=str(value["role"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
            logical_bytes=int(value["logical_bytes"]),
            checksum=str(value["checksum"]),
        )


@dataclass(frozen=True)
class SerializedCheckpoint:
    payload: bytes
    payload_checksum: str
    objects: tuple[CheckpointObject, ...]
    logical_bytes: int
    equivalence_checksum: str | None = None


@dataclass(frozen=True)
class TensorRedundancyEvidence:
    """A detector candidate edge anchored to one exact checkpoint tensor path."""

    rank: int
    role: str
    path: str
    local_index: int
    equivalent_locations: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "role": self.role,
            "path": self.path,
            "local_index": self.local_index,
            "equivalent_locations": [
                list(location) for location in self.equivalent_locations
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorRedundancyEvidence":
        return cls(
            rank=int(value["rank"]),
            role=str(value["role"]),
            path=str(value["path"]),
            local_index=int(value["local_index"]),
            equivalent_locations=tuple(
                (int(location[0]), int(location[1]))
                for location in value["equivalent_locations"]
            ),
        )


def snapshot_to_cpu(value: Any) -> Any:
    """Detach tensors into an immutable CPU snapshot while preserving structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, defaultdict):
        result = defaultdict(value.default_factory)
        result.update((key, snapshot_to_cpu(item)) for key, item in value.items())
        return result
    if isinstance(value, Mapping):
        items = ((key, snapshot_to_cpu(item)) for key, item in value.items())
        try:
            return value.__class__(items)
        except TypeError:
            return dict(items)
    if isinstance(value, list):
        return [snapshot_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        items = [snapshot_to_cpu(item) for item in value]
        if hasattr(value, "_fields"):
            return value.__class__(*items)
        return tuple(items)
    try:
        return copy.deepcopy(value)
    except (TypeError, RuntimeError):
        return value


def _component(value: Any) -> str:
    raw = str(value).replace("~", "~0").replace("/", "~1")
    return f"{type(value).__name__}:{raw}"


def checkpoint_path(components: tuple[Any, ...]) -> str:
    return "/" + "/".join(_component(item) for item in components)


def checkpoint_object_id(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]


def tensor_checksum(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    try:
        raw = cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError):
        buffer = io.BytesIO()
        torch.save(cpu_tensor, buffer, _use_new_zipfile_serialization=False)
        raw = buffer.getvalue()
    return hashlib.sha256(raw).hexdigest()


def checkpoint_state_checksum(value: Any) -> str:
    """Hash nested checkpoint state without serializing a second full payload."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(b"tensor:")
            digest.update(str(item.dtype).encode())
            digest.update(json.dumps(list(item.shape)).encode())
            digest.update(memoryview(tensor.numpy()))
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda entry: (type(entry).__name__, str(entry))):
                digest.update(pickle.dumps((type(key).__name__, key), protocol=5))
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"list[" if isinstance(item, list) else b"tuple(")
            for child in item:
                visit(child)
            digest.update(b"]" if isinstance(item, list) else b")")
        else:
            digest.update(pickle.dumps((type(item).__name__, item), protocol=5))

    visit(value)
    return digest.hexdigest()


def iter_checkpoint_objects(
    value: Any, *, include_checksums: bool = True
) -> Iterable[CheckpointObject]:
    """Yield deterministic tensor metadata from an arbitrary training state."""

    def visit(current: Any, components: tuple[Any, ...]) -> Iterable[CheckpointObject]:
        if isinstance(current, torch.Tensor):
            path = checkpoint_path(components)
            role = str(components[0]) if components else "root"
            logical_bytes = current.numel() * current.element_size()
            yield CheckpointObject(
                object_id=checkpoint_object_id(path),
                path=path,
                role=role,
                dtype=str(current.dtype),
                shape=tuple(current.shape),
                logical_bytes=logical_bytes,
                checksum=tensor_checksum(current) if include_checksums else "",
            )
            return
        if isinstance(current, Mapping):
            for key in sorted(current, key=lambda item: (type(item).__name__, str(item))):
                yield from visit(current[key], components + (key,))
            return
        if isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                yield from visit(item, components + (index,))

    yield from visit(value, ())


def serialize_checkpoint(
    state: Any,
    *,
    state_is_snapshot: bool = False,
    object_checksums: bool = True,
) -> SerializedCheckpoint:
    snapshot = state if state_is_snapshot else snapshot_to_cpu(state)
    objects = tuple(
        iter_checkpoint_objects(snapshot, include_checksums=object_checksums)
    )
    buffer = io.BytesIO()
    torch.save(snapshot, buffer, _use_new_zipfile_serialization=False)
    payload = buffer.getvalue()
    return SerializedCheckpoint(
        payload=payload,
        payload_checksum=hashlib.sha256(payload).hexdigest(),
        objects=objects,
        logical_bytes=sum(item.logical_bytes for item in objects),
    )


def deserialize_checkpoint(payload: bytes, expected_checksum: str | None = None) -> Any:
    checksum = hashlib.sha256(payload).hexdigest()
    if expected_checksum is not None and checksum != expected_checksum:
        raise ValueError(
            f"checkpoint payload checksum mismatch: {checksum} != {expected_checksum}"
        )
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def validate_checkpoint_objects(
    state: Any, expected: Iterable[CheckpointObject]
) -> None:
    expected_tuple = tuple(expected)
    checksum_flags = {bool(item.checksum) for item in expected_tuple}
    if len(checksum_flags) > 1:
        raise ValueError("checkpoint object catalog mixes checksum modes")
    actual = tuple(
        iter_checkpoint_objects(
            state,
            include_checksums=checksum_flags != {False},
        )
    )
    if actual != expected_tuple:
        raise ValueError("checkpoint object catalog validation failed")
