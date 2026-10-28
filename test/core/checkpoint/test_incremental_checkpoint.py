import copy
import hashlib

import pytest
import torch

from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
    capture_optimizer_update,
    replay_optimizer_updates,
)


BASE_ID = "sha256:" + hashlib.sha256(b"base-checkpoint").hexdigest()
BINDING = RecoveryBinding(
    rank=3,
    world_size=16,
    data_parallel_rank=1,
    pipeline_parallel_rank=1,
    tensor_parallel_rank=0,
)


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_equal(first, second)
    else:
        assert left == right


def _adam(parameter_groups):
    return torch.optim.AdamW(
        parameter_groups,
        lr=3e-3,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )


def _ids(optimizer):
    return [
        [
            f"group.{group_index}.parameter.{parameter_index}"
            for parameter_index, _ in enumerate(group["params"])
        ]
        for group_index, group in enumerate(optimizer.param_groups)
    ]


def _capture(optimizer, parameter_ids, epoch, metadata=None):
    return capture_optimizer_update(
        optimizer,
        parameter_ids=parameter_ids,
        binding=BINDING,
        base_id=BASE_ID,
        base_epoch=0,
        epoch=epoch,
        metadata=metadata,
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_ordered_updates_exactly_reconstruct_adamw_state(device):
    direct_parameters = [
        torch.nn.Parameter(torch.tensor([1.0, -2.0], device=device)),
        torch.nn.Parameter(torch.tensor([0.5], device=device)),
    ]
    direct = _adam(
        [
            {"params": [direct_parameters[0]], "weight_decay": 0.1},
            {
                "params": [direct_parameters[1]],
                "weight_decay": 0.0,
                "amsgrad": True,
            },
        ]
    )
    parameter_ids = _ids(direct)
    base_parameters = [parameter.detach().clone() for parameter in direct_parameters]
    base_optimizer = copy.deepcopy(direct.state_dict())
    records = []

    gradients = (
        (torch.tensor([0.25, -0.5]), None),
        (torch.tensor([-0.75, 0.125]), torch.tensor([0.5])),
        (torch.tensor([0.1, 0.2]), torch.tensor([-0.3])),
    )
    for epoch, values in enumerate(gradients, start=1):
        direct.param_groups[0]["lr"] = 3e-3 / epoch
        direct.param_groups[1]["lr"] = 2e-3 / epoch
        for parameter, gradient in zip(direct_parameters, values):
            parameter.grad = (
                None if gradient is None else gradient.to(device=device).clone()
            )
        records.append(
            _capture(
                direct,
                parameter_ids,
                epoch,
                metadata={"scheduler_step": epoch},
            )
        )
        direct.step()
        direct.zero_grad(set_to_none=True)

    replay_parameters = [
        torch.nn.Parameter(parameter.clone()) for parameter in base_parameters
    ]
    replay = _adam(
        [
            {"params": [replay_parameters[0]], "weight_decay": 0.1},
            {
                "params": [replay_parameters[1]],
                "weight_decay": 0.0,
                "amsgrad": True,
            },
        ]
    )
    replay.load_state_dict(base_optimizer)
    assert (
        replay_optimizer_updates(
            replay,
            records,
            parameter_ids=parameter_ids,
            binding=BINDING,
            base_id=BASE_ID,
            base_epoch=0,
        )
        == 3
    )

    for expected, actual in zip(direct_parameters, replay_parameters):
        assert torch.equal(expected, actual)
    _assert_equal(direct.state_dict(), replay.state_dict())


def test_cpu_replay_borrows_record_gradient_without_mutating_it(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = _adam([parameter])
    parameter_ids = _ids(optimizer)
    parameter.grad = torch.tensor([0.25, -0.5])
    record = _capture(optimizer, parameter_ids, 1)
    record_gradient = record["gradients"][0][0]
    before = record_gradient.clone()
    original_step = optimizer.step
    observed = {}

    def checked_step():
        observed["same_storage"] = parameter.grad.data_ptr() == record_gradient.data_ptr()
        return original_step()

    monkeypatch.setattr(optimizer, "step", checked_step)
    apply_optimizer_update(
        optimizer,
        record,
        expected_parameter_ids=parameter_ids,
        expected_binding=BINDING,
        expected_base_id=BASE_ID,
        expected_base_epoch=0,
        expected_epoch=1,
    )

    assert observed["same_storage"]
    assert torch.equal(record_gradient, before)
    assert parameter.grad is None


def test_bfloat16_gradient_record_replays_into_fp32_adamw():
    direct_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    direct = _adam([direct_parameter])
    parameter_ids = _ids(direct)
    direct_parameter.grad = torch.tensor([0.25, -0.5])
    record = capture_optimizer_update(
        direct,
        parameter_ids=parameter_ids,
        binding=BINDING,
        base_id=BASE_ID,
        base_epoch=0,
        epoch=1,
        gradient_dtype=torch.bfloat16,
    )
    direct.step()

    replay_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    replay = _adam([replay_parameter])
    apply_optimizer_update(
        replay,
        record,
        expected_parameter_ids=parameter_ids,
        expected_binding=BINDING,
        expected_base_id=BASE_ID,
        expected_base_epoch=0,
        expected_epoch=1,
    )

    assert record["gradient_dtype"] == "torch.bfloat16"
    assert record["gradients"][0][0].dtype == torch.bfloat16
    assert record["gradients"][0][0].untyped_storage().nbytes() == 4
    assert torch.equal(replay_parameter, direct_parameter)
    _assert_equal(replay.state_dict(), direct.state_dict())


def test_raw_bfloat16_gradient_plus_scale_replays_exact_fp32_adamw_step():
    raw_gradient = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    multiplier = 1.0 / 7.0
    direct_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    direct = _adam([direct_parameter])
    parameter_ids = _ids(direct)
    direct_parameter.grad = raw_gradient.float().mul_(multiplier)

    capture_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    capture = _adam([capture_parameter])
    capture_parameter.grad = raw_gradient.float()
    record = capture_optimizer_update(
        capture,
        parameter_ids=parameter_ids,
        binding=BINDING,
        base_id=BASE_ID,
        base_epoch=0,
        epoch=1,
        gradient_dtype=torch.bfloat16,
        gradient_multiplier=multiplier,
    )
    direct.step()

    replay_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    replay = _adam([replay_parameter])
    apply_optimizer_update(
        replay,
        record,
        expected_parameter_ids=parameter_ids,
        expected_binding=BINDING,
        expected_base_id=BASE_ID,
        expected_base_epoch=0,
        expected_epoch=1,
    )

    assert record["gradient_multiplier"] == multiplier
    assert torch.equal(replay_parameter, direct_parameter)
    _assert_equal(replay.state_dict(), direct.state_dict())


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("inf"), float("nan")])
def test_gradient_multiplier_must_be_finite_and_positive(multiplier):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = _adam([parameter])
    parameter.grad = torch.tensor([0.5])
    with pytest.raises(ValueError, match="gradient multiplier"):
        capture_optimizer_update(
            optimizer,
            parameter_ids=_ids(optimizer),
            binding=BINDING,
            base_id=BASE_ID,
            base_epoch=0,
            epoch=1,
            gradient_multiplier=multiplier,
        )


def test_gradient_record_rejects_unknown_or_mislabeled_encoding():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = _adam([parameter])
    parameter_ids = _ids(optimizer)
    parameter.grad = torch.tensor([0.5])
    with pytest.raises(ValueError, match="unsupported.*gradient dtype"):
        capture_optimizer_update(
            optimizer,
            parameter_ids=parameter_ids,
            binding=BINDING,
            base_id=BASE_ID,
            base_epoch=0,
            epoch=1,
            gradient_dtype=torch.float64,
        )

    record = _capture(optimizer, parameter_ids, 1)
    record["gradient_dtype"] = "torch.bfloat16"
    with pytest.raises(ValueError, match="shape or dtype"):
        apply_optimizer_update(
            optimizer,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=BINDING,
            expected_base_id=BASE_ID,
            expected_base_epoch=0,
            expected_epoch=1,
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_twenty_updates_crossing_scheduler_boundaries_replay_exactly(device):
    direct_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=device))
    direct = _adam([direct_parameter])
    direct_scheduler = torch.optim.lr_scheduler.StepLR(
        direct, step_size=7, gamma=0.9
    )
    parameter_ids = _ids(direct)
    for step in range(1, 6):
        direct_parameter.grad = torch.tensor(
            [step / 10, -step / 20], device=device
        )
        direct.step()
        direct_scheduler.step()
    base_parameter = direct_parameter.detach().clone()
    base_optimizer = copy.deepcopy(direct.state_dict())
    base_scheduler = copy.deepcopy(direct_scheduler.state_dict())

    records = []
    for epoch in range(1, 21):
        direct_parameter.grad = torch.tensor(
            [epoch / 100, -epoch / 200], device=device
        )
        records.append(_capture(direct, parameter_ids, epoch))
        direct.step()
        direct_scheduler.step()

    replay_parameter = torch.nn.Parameter(base_parameter.clone())
    replay = _adam([replay_parameter])
    replay_scheduler = torch.optim.lr_scheduler.StepLR(
        replay, step_size=7, gamma=0.9
    )
    replay.load_state_dict(base_optimizer)
    replay_scheduler.load_state_dict(base_scheduler)
    for epoch, record in enumerate(records, start=1):
        apply_optimizer_update(
            replay,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=BINDING,
            expected_base_id=BASE_ID,
            expected_base_epoch=0,
            expected_epoch=epoch,
        )
        replay_scheduler.step()

    assert torch.equal(direct_parameter, replay_parameter)
    _assert_equal(direct.state_dict(), replay.state_dict())
    assert direct_scheduler.state_dict() == replay_scheduler.state_dict()


def test_update_replay_rejects_holes_wrong_base_and_changed_topology():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = _adam([parameter])
    parameter_ids = _ids(optimizer)
    parameter.grad = torch.tensor([0.5])
    record = _capture(optimizer, parameter_ids, 5)
    record["base_epoch"] = 4

    with pytest.raises(ValueError, match="different base"):
        apply_optimizer_update(
            optimizer,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=BINDING,
            expected_base_id="sha256:" + "0" * 64,
            expected_base_epoch=4,
            expected_epoch=5,
        )
    with pytest.raises(ValueError, match="not contiguous"):
        apply_optimizer_update(
            optimizer,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=BINDING,
            expected_base_id=BASE_ID,
            expected_base_epoch=4,
            expected_epoch=6,
        )

    changed = _adam([torch.nn.Parameter(torch.ones(2))])
    with pytest.raises(ValueError, match="topology changed"):
        apply_optimizer_update(
            changed,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=BINDING,
            expected_base_id=BASE_ID,
            expected_base_epoch=4,
            expected_epoch=5,
        )


def test_same_shape_parameter_reorder_is_rejected():
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0])),
        torch.nn.Parameter(torch.tensor([2.0])),
    ]
    optimizer = _adam(parameters)
    parameter_ids = [["first", "second"]]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    record = _capture(optimizer, parameter_ids, 1)

    with pytest.raises(ValueError, match="identity or order"):
        apply_optimizer_update(
            optimizer,
            record,
            expected_parameter_ids=[["second", "first"]],
            expected_binding=BINDING,
            expected_base_id=BASE_ID,
            expected_base_epoch=0,
            expected_epoch=1,
        )


def test_corrupt_record_fails_before_optimizer_mutation():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = _adam([parameter])
    parameter_ids = _ids(optimizer)
    parameter.grad = torch.tensor([0.5])
    record = _capture(optimizer, parameter_ids, 1)
    before_parameter = parameter.detach().clone()
    before_options = copy.deepcopy(
        {
            key: value
            for key, value in optimizer.param_groups[0].items()
            if key != "params"
        }
    )
    record["gradients"][0][0] = torch.ones(2)

    with pytest.raises(ValueError, match="shape or dtype"):
        apply_optimizer_update(
            optimizer,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=BINDING,
            expected_base_id=BASE_ID,
            expected_base_epoch=0,
            expected_epoch=1,
        )

    assert torch.equal(parameter, before_parameter)
    assert optimizer.state == {}
    assert {
        key: value
        for key, value in optimizer.param_groups[0].items()
        if key != "params"
    } == before_options


def test_capture_rejects_unsupported_or_unbound_records():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(TypeError, match="AdamW only"):
        capture_optimizer_update(
            torch.optim.SGD([parameter], lr=1e-2),
            parameter_ids=[["weight"]],
            binding=BINDING,
            base_id=BASE_ID,
            base_epoch=0,
            epoch=1,
        )

    optimizer = torch.optim.AdamW([parameter], foreach=None)
    with pytest.raises(ValueError, match="foreach=False"):
        _capture(optimizer, [["weight"]], 1)
    optimizer = _adam([parameter])
    with pytest.raises(ValueError, match="base_id"):
        capture_optimizer_update(
            optimizer,
            parameter_ids=[["weight"]],
            binding=BINDING,
            base_id="not-a-checksum",
            base_epoch=0,
            epoch=1,
        )
    with pytest.raises(ValueError, match="newer"):
        capture_optimizer_update(
            optimizer,
            parameter_ids=[["weight"]],
            binding=BINDING,
            base_id=BASE_ID,
            base_epoch=2,
            epoch=2,
        )
    with pytest.raises(ValueError, match="finite JSON"):
        _capture(optimizer, [["weight"]], 1, metadata={"bad": torch.ones(1)})
