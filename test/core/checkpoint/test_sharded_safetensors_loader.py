import json

import torch
from safetensors.torch import save_file

from Merak.core.checkpoint import utils


class _ModelKeys:
    def state_dict(self):
        return {
            "0.weight": None,
            "0.bias": None,
            "1.model.weight": None,
        }


def test_sharded_loader_opens_each_matching_shard_once(tmp_path, monkeypatch):
    first = {"model.weight": torch.tensor([1]), "model.bias": torch.tensor([2])}
    second = {"model.model.weight": torch.tensor([3])}
    save_file(first, tmp_path / "first.safetensors")
    save_file(second, tmp_path / "second.safetensors")
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {
        "model.weight": "first.safetensors",
        "model.bias": "first.safetensors",
        "model.model.weight": "second.safetensors",
        "unused.weight": "missing.safetensors",
    }}))

    opened = []
    safe_open = utils.safe_open

    def count_open(path, **kwargs):
        opened.append(path)
        return safe_open(path, **kwargs)

    monkeypatch.setattr(utils, "safe_open", count_open)
    weights = utils.load_sharded_safetensors(_ModelKeys(), str(index_path))

    assert set(weights) == {"0.weight", "0.bias", "1.model.weight"}
    assert torch.equal(weights["0.weight"], first["model.weight"])
    assert torch.equal(weights["0.bias"], first["model.bias"])
    assert torch.equal(weights["1.model.weight"], second["model.model.weight"])
    assert opened == [str(tmp_path / "first.safetensors"), str(tmp_path / "second.safetensors")]
