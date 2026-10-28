"""Numerical comparison for checkpoint recovery state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


FP32_RTOL = 1e-6
FP32_ATOL = 1e-7
BF16_PAIRWISE_RELATIVE_BOUND = float(torch.finfo(torch.bfloat16).eps)
_COMPARISON_CHUNK_ELEMENTS = 8 * 1024 * 1024


def _bf16_ordered_bits(value: torch.Tensor) -> torch.Tensor:
    raw = value.contiguous().view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    return torch.where(raw.bitwise_and(0x8000).bool(), 0x10000 - raw, 0x8000 + raw)


def compare_checkpoint_state(actual: Any, expected: Any) -> dict[str, Any]:
    """Compare nested state using an explicit mixed-precision contract."""
    stats: dict[str, Any] = {
        "tensor_count": 0,
        "element_count": 0,
        "exact_mismatch_elements": 0,
        "non_tensor_mismatches": 0,
        "nonfinite_elements": 0,
        "max_abs_error": 0.0,
        "max_relative_error": 0.0,
        "max_bfloat16_ulp": 0,
        "max_bfloat16_tensor_relative_linf_error": 0.0,
        "structure_equal": True,
        "numerically_close": True,
    }

    def reject_structure() -> None:
        stats["structure_equal"] = False
        stats["numerically_close"] = False

    def visit(left: Any, right: Any) -> None:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                reject_structure()
                return
            if left.shape != right.shape or left.dtype != right.dtype:
                reject_structure()
                return
            left = left.detach().reshape(-1)
            right = right.detach().reshape(-1)
            stats["tensor_count"] += 1
            stats["element_count"] += left.numel()
            if not left.numel():
                return
            tensor_max_error = 0.0
            tensor_scale = 0.0
            for start in range(0, left.numel(), _COMPARISON_CHUNK_ELEMENTS):
                stop = min(start + _COMPARISON_CHUNK_ELEMENTS, left.numel())
                left_chunk = left[start:stop].cpu()
                right_chunk = right[start:stop].cpu()
                unequal = left_chunk != right_chunk
                stats["exact_mismatch_elements"] += int(unequal.sum().item())
                if not left.is_floating_point():
                    if unequal.any():
                        stats["numerically_close"] = False
                    continue
                finite = torch.isfinite(left_chunk) & torch.isfinite(right_chunk)
                stats["nonfinite_elements"] += int((~finite).sum().item())
                if not finite.all():
                    stats["numerically_close"] = False
                if not finite.any():
                    continue
                left_finite = left_chunk[finite].float()
                right_finite = right_chunk[finite].float()
                error = (left_finite - right_finite).abs()
                maximum_error = float(error.max().item())
                tensor_max_error = max(tensor_max_error, maximum_error)
                stats["max_abs_error"] = max(
                    stats["max_abs_error"], maximum_error
                )
                relative = error / right_finite.abs().clamp_min(
                    torch.finfo(torch.float32).tiny
                )
                stats["max_relative_error"] = max(
                    stats["max_relative_error"], float(relative.max().item())
                )
                if left.dtype == torch.bfloat16:
                    distance = (
                        _bf16_ordered_bits(left_chunk)
                        - _bf16_ordered_bits(right_chunk)
                    ).abs()
                    stats["max_bfloat16_ulp"] = max(
                        stats["max_bfloat16_ulp"], int(distance.max().item())
                    )
                    tensor_scale = max(
                        tensor_scale, float(right_finite.abs().max().item())
                    )
                elif not torch.allclose(
                    left_chunk, right_chunk, rtol=FP32_RTOL, atol=FP32_ATOL
                ):
                    stats["numerically_close"] = False
            if left.dtype == torch.bfloat16:
                relative_linf = (
                    tensor_max_error / tensor_scale
                    if tensor_scale > 0.0
                    else tensor_max_error
                )
                stats["max_bfloat16_tensor_relative_linf_error"] = max(
                    stats["max_bfloat16_tensor_relative_linf_error"],
                    relative_linf,
                )
                if relative_linf > BF16_PAIRWISE_RELATIVE_BOUND:
                    stats["numerically_close"] = False
            return
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if (
                not isinstance(left, Mapping)
                or not isinstance(right, Mapping)
                or left.keys() != right.keys()
            ):
                reject_structure()
                return
            for key in left:
                visit(left[key], right[key])
            return
        if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
            if (
                not isinstance(right, type(left))
                or len(left) != len(right)
            ):
                reject_structure()
                return
            for left_item, right_item in zip(left, right):
                visit(left_item, right_item)
            return
        if left != right:
            stats["non_tensor_mismatches"] += 1
            stats["numerically_close"] = False

    visit(actual, expected)
    stats["bitwise_equal"] = bool(
        stats["structure_equal"]
        and stats["exact_mismatch_elements"] == 0
        and stats["non_tensor_mismatches"] == 0
    )
    stats["contract"] = {
        "bfloat16_tensor_relative_linf_bound": BF16_PAIRWISE_RELATIVE_BOUND,
        "bfloat16_max_ulp": "reported_not_gated",
        "other_float_rtol": FP32_RTOL,
        "other_float_atol": FP32_ATOL,
        "nonfinite_allowed": False,
        "non_tensor_exact": True,
    }
    return stats
