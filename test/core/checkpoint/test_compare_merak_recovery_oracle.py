import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[3] / "tools" / "compare_merak_recovery_oracle.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_merak_recovery_oracle", MODULE_PATH
)
oracle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(oracle)


def test_logged_loss_tolerance_accepts_preregistered_llama_boundary():
    recovered = {
        11: "8.2021",
        12: "7.9833",
        13: "7.8192",
        14: "7.6371",
        15: "7.4928",
    }
    reference = {
        11: "8.2022",
        12: "7.9833",
        13: "7.8193",
        14: "7.6372",
        15: "7.4930",
    }

    result = oracle.compare_logged_losses(
        recovered,
        reference,
        expected_steps=range(11, 16),
        rtol=1e-4,
        atol=1e-4,
    )

    assert result["expected_steps_present"]
    assert result["all_finite"]
    assert not result["exact"]
    assert result["numerically_close"]
    assert result["max_absolute_error"] == pytest.approx(2e-4)
    assert result["max_relative_error"] < 3e-5


@pytest.mark.parametrize(
    ("recovered", "reference"),
    [
        ({11: "8.0", 12: "7.9"}, {11: "8.0", 12: "7.9", 13: "7.8"}),
        ({11: "8.0", 12: "7.9", 13: "7.8"}, {11: "8.0", 12: "7.9"}),
    ],
)
def test_logged_loss_contract_rejects_missing_expected_step(recovered, reference):
    result = oracle.compare_logged_losses(
        recovered,
        reference,
        expected_steps=(11, 12, 13),
        rtol=1e-4,
        atol=1e-4,
    )

    assert not result["expected_steps_present"]
    assert not result["numerically_close"]


@pytest.mark.parametrize("nonfinite", ["nan", "inf", "-inf"])
def test_logged_loss_contract_rejects_nonfinite_values(nonfinite):
    result = oracle.compare_logged_losses(
        {11: nonfinite},
        {11: "8.0"},
        expected_steps=(11,),
        rtol=1e-4,
        atol=1e-4,
    )

    assert not result["all_finite"]
    assert not result["numerically_close"]
    assert math.isinf(result["max_absolute_error"])


def test_logged_loss_contract_rejects_difference_outside_tolerance():
    result = oracle.compare_logged_losses(
        {11: "8.0020"},
        {11: "8.0000"},
        expected_steps=(11,),
        rtol=1e-4,
        atol=1e-4,
    )

    assert result["expected_steps_present"]
    assert result["all_finite"]
    assert not result["numerically_close"]


def test_logged_loss_contract_rejects_extra_recovered_step():
    result = oracle.compare_logged_losses(
        {10: "8.1", 11: "8.0"},
        {10: "8.1", 11: "8.0"},
        expected_steps=(11,),
        rtol=1e-4,
        atol=1e-4,
    )

    assert not result["recovered_steps_exact"]
    assert not result["expected_steps_present"]
    assert not result["numerically_close"]
