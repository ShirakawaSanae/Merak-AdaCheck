"""Gloo control plane for one locally-owned semantic optimizer update."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from os import PathLike
import time
from typing import Any, Mapping

import torch.distributed as dist

from .global_commit import manifest_evidence
from .matched import CONTROL_SOURCE_RANK
from .object_optimizer_update import ObjectOptimizerUpdate
from .object_semantic_plan import ActiveGradientPlan, ConfirmedGradientEncoding
from .object_update_transport import (
    ObjectUpdateLocator,
    ObjectUpdateTransferMetrics,
    PublishedObjectUpdate,
    build_local_object_update_bundles,
    commit_object_update_state,
    create_object_update_control,
    seal_object_update_host,
    write_object_update_locator,
)
from .replica_service import ReplicaServiceClient


def _require_group(group: object | None, plan: ActiveGradientPlan) -> tuple[int, int]:
    if group is None or not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("object update commit requires an explicit initialized group")
    if dist.get_backend(group) != "gloo":
        raise RuntimeError("object update commit requires a gloo group")
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    if (
        rank != dist.get_rank()
        or world_size != dist.get_world_size()
        or world_size != len(plan.rank_bindings)
        or {binding.rank for binding in plan.rank_bindings} != set(range(world_size))
        or {binding.host for binding in plan.rank_bindings} != {0, 1}
    ):
        raise ValueError("object update commit requires a world-size gloo group")
    return rank, world_size


def _local_evidence(
    *,
    rank: int,
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    update: ObjectOptimizerUpdate,
    clients: Mapping[int, ReplicaServiceClient],
    owned_payloads: bool = False,
) -> dict[str, Any]:
    if len(update.records) != 1 or update.records[0][0] != rank:
        raise ValueError("object update commit accepts only this rank's update")
    bindings = {binding.rank: binding for binding in plan.rank_bindings}
    target_ranks = {
        host: min(item.rank for item in plan.rank_bindings if item.host == host)
        for host in (0, 1)
    }
    bundles = {
        int(bundle["target_host"]): bundle
        for bundle in build_local_object_update_bundles(
            plan, encoding, update, owned_payloads=owned_payloads
        )
    }
    if set(bundles) != {0, 1}:
        raise ValueError("local object update did not build both host bundles")

    def commit_bundle(host: int) -> tuple[Any, int, float]:
        return commit_object_update_state(
            clients[host],
            bundles[host],
            source_rank=rank,
            source_host=bindings[rank].host,
            target_host=host,
            target_rank=target_ranks[host],
            group_ranks=tuple(range(len(plan.rank_bindings))),
            object_checksums=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = {host: executor.submit(commit_bundle, host) for host in (0, 1)}
        committed = {host: pending[host].result() for host in (0, 1)}

    manifests: dict[int, dict[str, Any]] = {}
    remote_payload_bytes = remote_wire_bytes = 0
    maximum_remote_put_ms = 0.0
    for host in (0, 1):
        manifest, wire_bytes, put_ms = committed[host]
        manifests[host] = manifest_evidence(manifest)
        if bindings[rank].host != host:
            remote_payload_bytes += manifest.payload_bytes
            remote_wire_bytes += wire_bytes
            maximum_remote_put_ms = max(maximum_remote_put_ms, put_ms)
    if set(manifests) != {0, 1}:
        raise ValueError("local object update did not commit both host bundles")
    return {
        "rank": rank,
        "bundles_by_host": [[host, manifests[host]] for host in (0, 1)],
        "metrics": {
            "remote_data_payload_bytes": remote_payload_bytes,
            "remote_wire_bytes": remote_wire_bytes,
            "maximum_remote_put_ms": maximum_remote_put_ms,
        },
    }


def _bundle_evidence(
    values: list[object], world_size: int
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    by_host = {0: [], 1: []}
    total_payload = total_wire = 0
    maximum_put_ms = 0.0
    ranks = set()
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "rank", "bundles_by_host", "metrics"
        }:
            raise ValueError("gathered object update evidence schema changed")
        rank = value["rank"]
        bundles, metrics = value["bundles_by_host"], value["metrics"]
        if (
            type(rank) is not int
            or rank in ranks
            or not isinstance(bundles, list)
            or not isinstance(metrics, Mapping)
            or set(metrics)
            != {"remote_data_payload_bytes", "remote_wire_bytes", "maximum_remote_put_ms"}
        ):
            raise ValueError("gathered object update evidence is invalid")
        ranks.add(rank)
        local = dict(bundles)
        if set(local) != {0, 1} or any(
            not isinstance(item, Mapping) for item in local.values()
        ):
            raise ValueError("gathered object update evidence lacks a host bundle")
        for host in (0, 1):
            evidence = dict(local[host])
            if int(evidence.get("source_rank", -1)) != rank:
                raise ValueError("gathered object update evidence has the wrong rank")
            by_host[host].append(evidence)
        payload, wire, put_ms = (
            metrics["remote_data_payload_bytes"],
            metrics["remote_wire_bytes"],
            metrics["maximum_remote_put_ms"],
        )
        if (
            type(payload) is not int
            or type(wire) is not int
            or not isinstance(put_ms, float)
        ):
            raise ValueError("gathered object update metrics are invalid")
        total_payload += payload
        total_wire += wire
        maximum_put_ms = max(maximum_put_ms, put_ms)
    if ranks != set(range(world_size)):
        raise ValueError("gathered object update evidence lacks a rank")
    return by_host, {
        "remote_data_payload_bytes": total_payload,
        "remote_wire_bytes": total_wire,
        "maximum_remote_put_ms": maximum_put_ms,
    }


def _publish(
    *,
    clients: Mapping[int, ReplicaServiceClient],
    run_id: str,
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    locator_path: str | PathLike[str],
    gathered: list[object],
    timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    evidence, totals = _bundle_evidence(gathered, len(plan.rank_bindings))
    evidence_finished = time.perf_counter()
    control = create_object_update_control(
        run_id=run_id,
        plan=plan,
        encoding=encoding,
        bundle_evidence_by_host=evidence,
    )
    target_ranks = {
        host: min(item.rank for item in plan.rank_bindings if item.host == host)
        for host in (0, 1)
    }
    controls = {}
    for host in (0, 1):
        manifest, wire_bytes, put_ms = commit_object_update_state(
            clients[host],
            control,
            source_rank=CONTROL_SOURCE_RANK,
            source_host=0,
            target_host=host,
            target_rank=target_ranks[host],
            group_ranks=tuple(range(len(plan.rank_bindings))),
            object_checksums=False,
        )
        controls[host] = manifest
        if host != 0:
            totals["remote_wire_bytes"] += wire_bytes
            totals["maximum_remote_put_ms"] = max(
                totals["maximum_remote_put_ms"], put_ms
            )
            totals["remote_control_payload_bytes"] = manifest.payload_bytes
    control_checksum = controls[0].payload_checksum
    if any(item.payload_checksum != control_checksum for item in controls.values()):
        raise ValueError("two hosts committed different object controls")
    control_finished = time.perf_counter()
    for host in (0, 1):
        seal_object_update_host(
            clients[host],
            epoch=encoding.epoch,
            control_checksum=control_checksum,
            control_evidence=manifest_evidence(controls[host]),
            bundle_evidence=evidence[host],
        )
    locator = ObjectUpdateLocator.create(
        run_id=run_id,
        plan=plan,
        encoding=encoding,
        control_payload_checksum=control_checksum,
        control_manifests_by_host=controls,
    )
    write_object_update_locator(locator_path, locator)
    if timing is not None:
        timing.update(
            {
                "coordinator_evidence_ms": (evidence_finished - started) * 1000.0,
                "coordinator_control_commit_ms": (control_finished - evidence_finished)
                * 1000.0,
                "coordinator_seal_locator_ms": (time.perf_counter() - control_finished)
                * 1000.0,
            }
        )
    return {
        "locator": locator.to_dict(),
        "metrics": {
            "remote_gradient_logical_bytes": encoding.total_remote_logical_bytes,
            "remote_data_payload_bytes": totals["remote_data_payload_bytes"],
            "remote_control_payload_bytes": totals.get(
                "remote_control_payload_bytes", 0
            ),
            "remote_wire_bytes": totals["remote_wire_bytes"],
            "maximum_remote_put_ms": totals["maximum_remote_put_ms"],
        },
    }


def commit_distributed_object_update_epoch(
    clients: Mapping[int, ReplicaServiceClient],
    *,
    run_id: str,
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    update: ObjectOptimizerUpdate,
    locator_path: str | PathLike[str],
    group: object | None,
    owned_payloads: bool = False,
) -> PublishedObjectUpdate:
    """Commit one rank-owned update through the two-host sealed protocol."""
    rank, world_size = _require_group(group, plan)
    if set(clients) != {0, 1} or not run_id:
        raise ValueError("object update commit requires two host clients and a run id")
    local_started = time.perf_counter()
    local: dict[str, Any]
    try:
        local = {
            "ok": True,
            "evidence": _local_evidence(
                rank=rank,
                plan=plan,
                encoding=encoding,
                update=update,
                clients=clients,
                owned_payloads=owned_payloads,
            ),
        }
    except Exception as error:
        local = {
            "ok": False,
            "error": f"rank {rank} {type(error).__name__}: {error}",
        }
    local_finished = time.perf_counter()
    gathered: list[object] = [None] * world_size
    gather_started = time.perf_counter()
    dist.all_gather_object(gathered, local, group=group)
    gather_finished = time.perf_counter()
    response: object = None
    coordinator_timing: dict[str, float] = {}
    if rank == 0:
        try:
            failures = [
                item
                for item in gathered
                if not isinstance(item, Mapping) or not item.get("ok")
            ]
            if failures:
                raise RuntimeError(str(failures[0].get("error", "rank commit failed")))
            response = {
                "ok": True,
                "published": _publish(
                    clients=clients,
                    run_id=run_id,
                    plan=plan,
                    encoding=encoding,
                    locator_path=locator_path,
                    gathered=[item["evidence"] for item in gathered],
                    timing=coordinator_timing,
                ),
            }
        except Exception as error:
            response = {
                "ok": False,
                "error": f"rank 0 {type(error).__name__}: {error}",
            }
    broadcast = [response]
    broadcast_started = time.perf_counter()
    dist.broadcast_object_list(broadcast, src=0, group=group)
    broadcast_finished = time.perf_counter()
    value = broadcast[0]
    if not isinstance(value, Mapping) or set(value) != (
        {"ok", "published"} if value.get("ok") else {"ok", "error"}
    ):
        raise RuntimeError("object update commit broadcast schema changed")
    if not value["ok"]:
        raise RuntimeError(str(value["error"]))
    published = value["published"]
    if not isinstance(published, Mapping) or set(published) != {"locator", "metrics"}:
        raise RuntimeError("object update commit publication schema changed")
    locator = ObjectUpdateLocator.from_dict(published["locator"])
    metrics = published["metrics"]
    expected = {
        "remote_gradient_logical_bytes",
        "remote_data_payload_bytes",
        "remote_control_payload_bytes",
        "remote_wire_bytes",
        "maximum_remote_put_ms",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected:
        raise RuntimeError("object update commit metrics schema changed")
    result = PublishedObjectUpdate(
        locator,
        ObjectUpdateTransferMetrics(
            int(metrics["remote_gradient_logical_bytes"]),
            int(metrics["remote_data_payload_bytes"]),
            int(metrics["remote_control_payload_bytes"]),
            int(metrics["remote_wire_bytes"]),
            float(metrics["maximum_remote_put_ms"]),
        ),
        {
            "local_bundle_commit_ms": (local_finished - local_started) * 1000.0,
            "manifest_gather_ms": (gather_finished - gather_started) * 1000.0,
            "coordinator_evidence_ms": coordinator_timing.get(
                "coordinator_evidence_ms"
            ),
            "coordinator_control_commit_ms": coordinator_timing.get(
                "coordinator_control_commit_ms"
            ),
            "coordinator_seal_locator_ms": coordinator_timing.get(
                "coordinator_seal_locator_ms"
            ),
            "result_broadcast_ms": (broadcast_finished - broadcast_started) * 1000.0,
        },
    )
    if (
        result.locator.epoch != encoding.epoch
        or result.locator.world_size != world_size
        or result.locator.run_id != run_id
        or result.locator.arm != plan.arm
        or result.locator.detector_catalog_hash != plan.detector_catalog_hash
        or result.locator.semantic_plan_hash != plan.semantic_plan_hash
        or result.locator.plan_hash != plan.plan_hash
        or result.locator.canonical_hash != plan.canonical_hash
        or result.locator.encoding_hash != encoding.encoding_hash
    ):
        raise RuntimeError("broadcast object update locator disagrees with this rank")
    return result


def commit_owned_snapshot_object_update_epoch(
    clients: Mapping[int, ReplicaServiceClient],
    *,
    run_id: str,
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    update: ObjectOptimizerUpdate,
    locator_path: str | PathLike[str],
    group: object | None,
) -> PublishedObjectUpdate:
    """Commit an update whose CPU snapshot remains private until this call returns."""
    return commit_distributed_object_update_epoch(
        clients,
        run_id=run_id,
        plan=plan,
        encoding=encoding,
        update=update,
        locator_path=locator_path,
        group=group,
        owned_payloads=True,
    )
