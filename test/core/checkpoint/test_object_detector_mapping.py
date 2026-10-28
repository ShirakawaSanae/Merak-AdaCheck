import hashlib
import json

import pytest
import torch

from Merak.core.checkpoint.object_detector_mapping import (
    map_detector_catalog_to_gradient_shards,
)
from Merak.core.checkpoint.object_semantic_plan import RankBinding


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _catalog(tmp_path, requirements, *, role="optimizer"):
    item = {
        "value_id": "1" * 64,
        "role": role,
        "dtype": "torch.float16",
        "shape": [2, 3],
        "logical_bytes": 12,
        "checksum": "2" * 64,
        "source_ranks": sorted({item["rank"] for item in requirements}),
        "source_hosts": ["host0", "host1"],
        "requirements": sorted(requirements, key=lambda item: (item["rank"], item["path"])),
    }
    path = tmp_path / "object_catalog.jsonl"
    path.write_text(json.dumps(item, sort_keys=True) + "\n")
    canonical = {
        "full_logical_bytes": 12 * len(requirements),
        "template_ranks": [0, 1],
        "objects": [
            {
                key: item[key]
                for key in (
                    "value_id",
                    "role",
                    "dtype",
                    "shape",
                    "logical_bytes",
                    "checksum",
                    "source_ranks",
                    "requirements",
                )
            }
        ],
    }
    return path, _digest(canonical)


def _requirement(rank, group=0, index=0):
    return {
        "rank": rank,
        "path": f"/str:optimizer/str:fp16_groups/int:{group}/int:{index}",
        "object_id": f"object-{rank}",
        "alias_id": f"alias-{rank}",
    }


def _map(path, catalog_hash, rank, parameter_ids, parameters, **kwargs):
    return map_detector_catalog_to_gradient_shards(
        path,
        expected_catalog_hash=catalog_hash,
        world_size=2,
        binding=RankBinding(rank, 0, kwargs.get("tensor_rank", 0), rank, rank),
        parameter_ids=parameter_ids,
        parameter_groups=parameters,
        tensor_parallel_rank=kwargs.get("tensor_rank", 0),
        tensor_parallel_world_size=kwargs.get("tensor_world", 1),
        encoded_dtype=torch.bfloat16,
    )


def test_detector_class_is_shared_across_dp_and_missing_slot_is_rank_local(tmp_path):
    path, catalog_hash = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    rank0 = _map(path, catalog_hash, 0, (("weight", "bias"),), ((parameter, parameter),))
    rank1 = _map(path, catalog_hash, 1, (("weight",),), ((parameter,),))

    assert rank0.shards[0].detector_class_id == rank1.shards[0].detector_class_id
    assert rank0.detector_mapped == 1
    assert rank0.rank_local_fallback == 1
    assert rank0.records[1].source == "rank_local_fallback"
    assert rank0.records[1].detector_class_id != rank1.records[0].detector_class_id
    assert rank0.mapping_hash == _map(
        path, catalog_hash, 0, (("weight", "bias"),), ((parameter, parameter),)
    ).mapping_hash


def test_catalog_tamper_and_local_bad_group_index_fail_closed(tmp_path):
    path, catalog_hash = _catalog(tmp_path, [_requirement(0, group=1), _requirement(1)])
    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="outside live parameter groups"):
        _map(path, catalog_hash, 0, (("weight",),), ((parameter,),))

    path, catalog_hash = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    item = json.loads(path.read_text())
    item["logical_bytes"] += 1
    path.write_text(json.dumps(item, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        _map(path, catalog_hash, 0, (("weight",),), ((parameter,),))


def test_catalog_schema_duplicate_requirement_and_world_rank_fail_closed(tmp_path):
    path, catalog_hash = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    item = json.loads(path.read_text())
    item["source_hosts"] = ["host1", "host0"]
    path.write_text(json.dumps(item, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="invalid metadata"):
        _map(path, catalog_hash, 0, (("weight",),), ((torch.zeros(2, 3),),))

    path, catalog_hash = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    item = json.loads(path.read_text())
    item["requirements"].insert(1, dict(item["requirements"][0]))
    item["source_ranks"] = [0, 1]
    path.write_text(json.dumps(item, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="duplicate detector requirement"):
        _map(path, catalog_hash, 0, (("weight",),), ((torch.zeros(2, 3),),))

    path, catalog_hash = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    item = json.loads(path.read_text())
    item["requirements"][1]["rank"] = 2
    item["source_ranks"] = [0, 2]
    path.write_text(json.dumps(item, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="requirement is invalid"):
        _map(path, catalog_hash, 0, (("weight",),), ((torch.zeros(2, 3),),))


def test_only_optimizer_fp16_group_paths_map_to_detector_classes(tmp_path):
    path, catalog_hash = _catalog(
        tmp_path, [_requirement(0), _requirement(1)], role="model"
    )
    result = _map(
        path, catalog_hash, 0, (("weight",),), ((torch.zeros(2, 3),),)
    )
    assert result.detector_mapped == 0
    assert result.rank_local_fallback == 1
    assert result.records[0].source == "rank_local_fallback"


def test_scalar_optimizer_metadata_is_valid_but_not_a_gradient_slot(tmp_path):
    path, _ = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    scalar = json.loads(path.read_text())
    scalar.update(
        value_id="0" * 64,
        shape=[],
        logical_bytes=4,
        requirements=[
            {
                **_requirement(rank),
                "path": f"/str:optimizer/str:state/int:{rank}/str:step",
            }
            for rank in (0, 1)
        ],
    )
    scalar["source_ranks"] = [0, 1]
    tensor = json.loads(path.read_text())
    objects = [scalar, tensor]
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in objects))
    catalog_hash = _digest(
        {
            "full_logical_bytes": 4 * 2 + 12 * 2,
            "template_ranks": [0, 1],
            "objects": [
                {
                    key: item[key]
                    for key in (
                        "value_id", "role", "dtype", "shape", "logical_bytes",
                        "checksum", "source_ranks", "requirements",
                    )
                }
                for item in objects
            ],
        }
    )
    result = _map(
        path, catalog_hash, 0, (("weight",),), ((torch.zeros(2, 3),),)
    )
    assert result.detector_mapped == 1


def test_contiguous_tp_offsets_and_missing_metadata_rejection(tmp_path):
    path, catalog_hash = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    parameter.model_parallel = True
    parameter.partition_dim = 1
    parameter.partition_stride = 1
    result = _map(
        path,
        catalog_hash,
        1,
        (("weight",),),
        ((parameter,),),
        tensor_rank=1,
        tensor_world=2,
    )
    assert result.shards[0].global_shape == (2, 6)
    assert result.shards[0].offsets == (0, 3)
    assert result.shards[0].lengths == (2, 3)

    del parameter.partition_dim
    with pytest.raises(ValueError, match="lacks TP metadata"):
        _map(
            path,
            catalog_hash,
            1,
            (("weight",),),
            ((parameter,),),
            tensor_rank=1,
            tensor_world=2,
        )

    parameter.partition_dim = 1
    parameter.partition_stride = 2
    with pytest.raises(ValueError, match="contiguous tensor-parallel"):
        _map(
            path,
            catalog_hash,
            1,
            (("weight",),),
            ((parameter,),),
            tensor_rank=1,
            tensor_world=2,
        )


def test_pp_local_model_parallel_marker_needs_no_tp_metadata_at_tp_one(tmp_path):
    path, catalog_hash = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    parameter.model_parallel = True
    parameter.pipeline_parallel = True

    result = _map(
        path,
        catalog_hash,
        0,
        (("pp_local.weight",),),
        ((parameter,),),
        tensor_rank=0,
        tensor_world=1,
    )

    assert result.shards[0].global_shape == (2, 3)
    assert result.shards[0].offsets == (0, 0)
    assert not hasattr(parameter, "partition_dim")


def test_pipeline_blanket_marker_is_replicated_and_legacy_bias_stride_is_supported(tmp_path):
    path, catalog_hash = _catalog(tmp_path, [_requirement(0), _requirement(1)])
    replicated = torch.nn.Parameter(torch.zeros(2, 3))
    replicated.model_parallel = True
    result = _map(
        path,
        catalog_hash,
        1,
        (("layer_norm.weight",),),
        ((replicated,),),
        tensor_rank=1,
        tensor_world=2,
    )
    assert result.shards[0].global_shape == (2, 3)
    assert result.shards[0].offsets == (0, 0)

    bias = torch.nn.Parameter(torch.zeros(3))
    bias.model_parallel = True
    bias.partition_dim = 0
    bias.stride = 1
    result = _map(
        path,
        catalog_hash,
        1,
        (("projection.bias",),),
        ((bias,),),
        tensor_rank=1,
        tensor_world=2,
    )
    assert result.shards[0].global_shape == (6,)
    assert result.shards[0].offsets == (3,)
