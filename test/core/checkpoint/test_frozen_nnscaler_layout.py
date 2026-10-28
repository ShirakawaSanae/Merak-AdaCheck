import json
from pathlib import Path

import pytest

from examples.checkpoint_grouping.export_frozen_nnscaler_layout import build_layout


def _generated_code(root: Path, *, invalid_slice: bool = False) -> Path:
    root.mkdir(parents=True)
    for rank in range(16):
        start, stop = ((0, 2) if rank < 8 else (2, 4))
        if invalid_slice and rank == 0:
            stop = 5
        root.joinpath(f"gencode{rank}.py").write_text(
            "class Generated:\n"
            "    def __init__(self):\n"
            f"        self.register_parameter('weight_{rank}', torch.nn.Parameter(torch.empty(({stop - start},), dtype=torch.float32)))\n"
            f"        self.add_full_map('weight_{rank}', 1, True, 'model.weight', (4,), (slice({start}, {stop}, None),), 1)\n"
            f"        self.register_buffer('buffer_{rank}', torch.empty((1,), dtype=torch.float32))\n"
            f"        self.add_full_map('buffer_{rank}', 2, False, 'model.buffer', (1,), (slice(0, 1, None),), 1)\n"
            "        self.init_group(ranks=[0, 1])\n"
        )
    return root


def test_freezes_deterministic_16_rank_parameter_layout(tmp_path):
    generated = _generated_code(tmp_path / "gds4")

    first = build_layout(generated, name="gds4", model_name="LLaMA-3-8B")
    second = build_layout(generated, name="gds4", model_name="LLaMA-3-8B")

    assert first["layout_hash"] == second["layout_hash"]
    assert first["world_size"] == 16
    assert first["rank_placement"][8] == {
        "rank": 8,
        "host_index": 1,
        "host_id": "1",
        "local_device": 0,
    }
    assert first["parameters"] == [
        {
            "original_name": "model.weight",
            "full_shape": (4,),
            "dtype": "torch.float32",
            "holder_ranks": list(range(16)),
            "shards": first["parameters"][0]["shards"],
            "coverage_validation": "slice_bounds_only",
        }
    ]
    assert len(first["parameters"][0]["shards"]) == 16
    assert first["generated_process_groups"] == [[0, 1]]
    assert first["optimizer_layout"]["complete"] is False
    assert first["online_runtime_eligible"] is False
    assert first["provenance"]["imports_nnscaler"] is False
    json.dumps(first, sort_keys=True)


def test_rejects_out_of_bounds_generated_slice(tmp_path):
    generated = _generated_code(tmp_path / "gds8", invalid_slice=True)

    with pytest.raises(ValueError, match="outside the full shape"):
        build_layout(generated, name="gds8", model_name="LLaMA-3-8B")


def test_requires_every_rank_gencode_file(tmp_path):
    generated = _generated_code(tmp_path / "gds4")
    generated.joinpath("gencode15.py").unlink()

    with pytest.raises(ValueError, match="expected gencode0.py"):
        build_layout(generated, name="gds4", model_name="LLaMA-3-8B")
