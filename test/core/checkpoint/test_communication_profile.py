import json

import pytest

from Merak.core.checkpoint.communication_profile import (
    CommunicationProfile,
    clip_sort_merge,
    directional_idle_slack_ns,
    idle_slack_ns,
    quantile,
    summarize_tail_ns,
)


def test_clip_sort_merge_clips_and_coalesces_adjacent_intervals():
    assert clip_sort_merge([(30, 60), (-5, 10), (10, 20), (55, 120)], 0, 100) == [
        (0, 20),
        (30, 100),
    ]


def test_idle_slack_uses_busy_union_and_directional_union():
    assert idle_slack_ns(0, 100, [(5, 15), (30, 40), (60, 80)]) == 60
    assert idle_slack_ns(0, 100, [(5, 15), (30, 40), (60, 80)], min_gap_ns=15) == 55
    assert directional_idle_slack_ns(0, 100, [(5, 15), (30, 40)], [(10, 20), (60, 80)]) == 55


def test_quantiles_and_tail_summary_are_deterministic():
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    assert quantile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert summarize_tail_ns([])["p99_ns"] is None

    summary = summarize_tail_ns([1, 2, 3, 4])
    assert summary == {
        "count": 4,
        "mean_ns": 2.5,
        "min_ns": 1,
        "p50_ns": 2.5,
        "p95_ns": pytest.approx(3.85),
        "p99_ns": pytest.approx(3.97),
        "max_ns": 4,
    }


def test_disabled_profile_has_no_file_or_timestamp_validation(tmp_path):
    path = tmp_path / "disabled.jsonl"
    profile = CommunicationProfile(enabled=False, jsonl_path=path)

    assert profile.record_interval("ignored", 2, 1) is False
    assert profile.record_event("ignored", "not-a-timestamp") is False
    assert profile.events == ()
    assert not path.exists()


def test_profile_writes_compact_jsonl_and_optionally_retains_records(tmp_path):
    path = tmp_path / "nested" / "events.jsonl"
    with CommunicationProfile(enabled=True, jsonl_path=path, retain_events=True) as profile:
        assert profile.record_event("enqueue", 101, rank=3)
        assert profile.record_interval("remote_edge", 110, 150, source=3, destination=11)
        assert len(profile.events) == 2

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {
            "at_ns": 101,
            "fields": {"rank": 3},
            "kind": "event",
            "name": "enqueue",
            "schema_version": 1,
        },
        {
            "duration_ns": 40,
            "end_ns": 150,
            "fields": {"destination": 11, "source": 3},
            "kind": "interval",
            "name": "remote_edge",
            "schema_version": 1,
            "start_ns": 110,
        },
    ]
