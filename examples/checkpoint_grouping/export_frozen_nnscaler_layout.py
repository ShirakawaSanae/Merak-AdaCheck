#!/usr/bin/env python3
"""Freeze static parameter placement from existing nnScaler generated code.

This parser intentionally uses only the Python standard library.  It consumes
``gencode<N>.py`` files as immutable input and does not import nnScaler, load
model weights, or infer optimizer shards.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_GENCODE = re.compile(r"gencode(\d+)\.py$")
_LABEL = re.compile(r"[A-Za-z0-9_.-]+$")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _literal(node: ast.AST, description: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{description} must be a literal") from error


def _attribute_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _shape(node: ast.AST, description: str) -> tuple[int, ...]:
    value = _literal(node, description)
    if not isinstance(value, tuple) or any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
        for dimension in value
    ):
        raise ValueError(f"{description} must be a tuple of non-negative integers")
    return tuple(value)


def _slices(node: ast.AST, shape: tuple[int, ...], description: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(node, ast.Tuple) or len(node.elts) != len(shape):
        raise ValueError(f"{description} dimensionality does not match full shape")
    ranges = []
    for dimension, slice_node in zip(shape, node.elts):
        if not (
            isinstance(slice_node, ast.Call)
            and isinstance(slice_node.func, ast.Name)
            and slice_node.func.id == "slice"
            and len(slice_node.args) == 3
            and not slice_node.keywords
        ):
            raise ValueError(f"{description} contains an unsupported slice")
        start, stop, step = (
            _literal(argument, description) for argument in slice_node.args
        )
        if start is None:
            start = 0
        if stop is None:
            stop = dimension
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
            or step not in (None, 1)
            or start < 0
            or stop < start
            or stop > dimension
        ):
            raise ValueError(f"{description} is outside the full shape")
        ranges.append((start, stop))
    return tuple(ranges)


def _registered_tensors(tree: ast.AST, source: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"register_parameter", "register_buffer"}
            and len(node.args) >= 2
        ):
            continue
        local_name = _literal(node.args[0], f"{source}: registered name")
        if not isinstance(local_name, str):
            raise ValueError(f"{source}: registered name must be a string")
        tensor_call = next(
            (
                descendant
                for descendant in ast.walk(node.args[1])
                if isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Attribute)
                and descendant.func.attr == "empty"
            ),
            None,
        )
        if tensor_call is None or not tensor_call.args:
            raise ValueError(f"{source}: cannot read tensor shape for {local_name}")
        shape = _shape(tensor_call.args[0], f"{source}: local shape for {local_name}")
        dtype = None
        for keyword in tensor_call.keywords:
            if keyword.arg == "dtype":
                dtype = _attribute_name(keyword.value)
        if dtype is None:
            raise ValueError(f"{source}: cannot read dtype for {local_name}")
        record = {
            "kind": "parameter" if node.func.attr == "register_parameter" else "buffer",
            "local_shape": list(shape),
            "dtype": dtype,
        }
        previous = records.setdefault(local_name, record)
        if previous != record:
            raise ValueError(f"{source}: conflicting registrations for {local_name}")
    return records


def _maps_for_file(path: Path, rank: int) -> tuple[list[dict[str, Any]], list[tuple[int, ...]]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    registered = _registered_tensors(tree, path)
    mappings = []
    process_groups = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "init_group":
            ranks_node = node.args[0] if node.args else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "ranks"),
                None,
            )
            if ranks_node is None:
                continue
            ranks = _literal(ranks_node, f"{path}: init_group ranks")
            if (
                isinstance(ranks, list)
                and all(isinstance(item, int) and not isinstance(item, bool) for item in ranks)
            ):
                process_groups.append(tuple(ranks))
            continue
        if node.func.attr != "add_full_map":
            continue
        if len(node.args) != 7 or node.keywords:
            raise ValueError(f"{path}: add_full_map must have seven positional arguments")
        local_name = _literal(node.args[0], f"{path}: local map name")
        is_parameter = _literal(node.args[2], f"{path}: parameter flag")
        original_name = _literal(node.args[3], f"{path}: original parameter name")
        if not isinstance(local_name, str) or not isinstance(is_parameter, bool) or not isinstance(original_name, str):
            raise ValueError(f"{path}: unsupported add_full_map identity")
        registration = registered.get(local_name)
        if registration is None:
            raise ValueError(f"{path}: map has no registered tensor: {local_name}")
        expected_kind = "parameter" if is_parameter else "buffer"
        if registration["kind"] != expected_kind:
            raise ValueError(f"{path}: map/register kind mismatch for {local_name}")
        full_shape = _shape(node.args[4], f"{path}: full shape for {original_name}")
        index_ranges = _slices(node.args[5], full_shape, f"{path}: slice for {original_name}")
        local_shape = tuple(stop - start for start, stop in index_ranges)
        if tuple(registration["local_shape"]) != local_shape:
            raise ValueError(f"{path}: local shape disagrees with slice for {local_name}")
        mappings.append(
            {
                "rank": rank,
                "local_name": local_name,
                "original_name": original_name,
                "kind": expected_kind,
                "dtype": registration["dtype"],
                "full_shape": list(full_shape),
                "index_ranges": [list(item) for item in index_ranges],
                "local_shape": list(local_shape),
            }
        )
    if not mappings:
        raise ValueError(f"{path}: contains no add_full_map records")
    return mappings, process_groups


def _gencode_files(generated_dir: Path, world_size: int) -> list[tuple[int, Path]]:
    files = []
    for path in generated_dir.glob("gencode*.py"):
        match = _GENCODE.fullmatch(path.name)
        if match is None:
            continue
        files.append((int(match.group(1)), path))
    files.sort()
    ranks = [rank for rank, _ in files]
    if ranks != list(range(world_size)):
        raise ValueError(
            f"{generated_dir}: expected gencode0.py through gencode{world_size - 1}.py"
        )
    return files


def _parameter_map(mappings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mapping in mappings:
        if mapping["kind"] == "parameter":
            by_name[str(mapping["original_name"])].append(mapping)
    parameters = []
    for name, shards in sorted(by_name.items()):
        shapes = {tuple(item["full_shape"]) for item in shards}
        dtypes = {item["dtype"] for item in shards}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError(f"conflicting full shape or dtype for {name}")
        shards.sort(
            key=lambda item: (item["rank"], item["index_ranges"], item["local_name"])
        )
        parameters.append(
            {
                "original_name": name,
                "full_shape": next(iter(shapes)),
                "dtype": next(iter(dtypes)),
                "holder_ranks": sorted({int(item["rank"]) for item in shards}),
                "shards": shards,
                "coverage_validation": "slice_bounds_only",
            }
        )
    if not parameters:
        raise ValueError("generated code contains no model parameter mappings")
    return parameters


def build_layout(
    generated_dir: Path,
    *,
    name: str,
    model_name: str,
    world_size: int = 16,
    ranks_per_host: int = 8,
    host_ids: tuple[str, ...] = ("17", "1"),
) -> dict[str, Any]:
    """Return a validated, deterministic static placement artifact."""
    generated_dir = generated_dir.resolve()
    if world_size != 16 or ranks_per_host != 8:
        raise ValueError("this exporter is restricted to 16 ranks on two 8-GPU hosts")
    if len(host_ids) != world_size // ranks_per_host or len(set(host_ids)) != len(host_ids):
        raise ValueError("host IDs must name the two distinct physical hosts")
    files = _gencode_files(generated_dir, world_size)
    mappings = []
    process_groups = set()
    file_digests = []
    for rank, path in files:
        contents = path.read_bytes()
        file_digests.append({"name": path.name, "sha256": _sha256_bytes(contents)})
        file_maps, file_groups = _maps_for_file(path, rank)
        mappings.extend(file_maps)
        process_groups.update(tuple(group) for group in file_groups)
    for group in process_groups:
        if any(rank < 0 or rank >= world_size for rank in group):
            raise ValueError(f"{generated_dir}: process group contains an out-of-range rank")
    mappings.sort(
        key=lambda item: (
            item["original_name"], item["rank"], item["index_ranges"], item["local_name"]
        )
    )
    parameters = _parameter_map(mappings)
    rank_placement = [
        {
            "rank": rank,
            "host_index": rank // ranks_per_host,
            "host_id": host_ids[rank // ranks_per_host],
            "local_device": rank % ranks_per_host,
        }
        for rank in range(world_size)
    ]
    identity = {
        "schema_version": 1,
        "name": name,
        "model_name": model_name,
        "world_size": world_size,
        "ranks_per_host": ranks_per_host,
        "rank_placement": rank_placement,
        "parameters": parameters,
        "state_maps": mappings,
        "generated_process_groups": [list(group) for group in sorted(process_groups)],
        "optimizer_layout": {
            "complete": False,
            "reason": "add_full_map exposes model parameter/buffer maps, not optimizer shards",
        },
        "online_runtime_eligible": False,
    }
    layout_hash = _sha256_bytes(_canonical_bytes(identity))
    return {
        **identity,
        "layout_hash": layout_hash,
        "provenance": {
            "producer": "nnscaler generated-code static parser",
            "imports_nnscaler": False,
            "generated_dir": str(generated_dir),
            "gencode_files": file_digests,
            "gencode_source_hash": _sha256_bytes(_canonical_bytes(file_digests)),
            "weights_copied": False,
        },
        "capabilities": {
            "static_model_parameter_placement": True,
            "offline_detector_catalog_input": True,
            "optimizer_layout_complete": False,
            "online_runtime_eligible": False,
        },
        "limitations": [
            "Only add_full_map model parameters and generated buffers are exported.",
            "Optimizer, gradient, RNG, scheduler, and runtime-state placement are not represented.",
            "Slice validation checks bounds and local shapes, not semantic tensor coverage.",
            "Generated process groups have no preserved PP/TP/DP semantic labels.",
            "This static layout cannot be passed to the current mirror-DP Merak incremental runtime.",
        ],
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    temporary.replace(path)


def _input(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not _LABEL.fullmatch(name) or not raw_path:
        raise argparse.ArgumentTypeError("--input must be NAME=/path/to/generated-code")
    return name, Path(raw_path)


def _host_ids(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(result) != 2 or len(set(result)) != 2:
        raise argparse.ArgumentTypeError("--host-ids must contain two distinct IDs")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=_input,
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="An explicit gds4/gds8 generated-code directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="LLaMA-3-8B")
    parser.add_argument("--host-ids", type=_host_ids, default=("17", "1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [name for name, _ in args.input]
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("exactly two uniquely named generated-code inputs are required")
    for name, generated_dir in args.input:
        artifact = build_layout(
            generated_dir,
            name=name,
            model_name=args.model_name,
            host_ids=args.host_ids,
        )
        output = args.output_dir / f"{name}.json"
        _atomic_json(output, artifact)
        print(json.dumps({"name": name, "layout_hash": artifact["layout_hash"], "output": str(output)}))


if __name__ == "__main__":
    main()
