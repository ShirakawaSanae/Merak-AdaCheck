import multiprocessing
import os

import pytest

from Merak.core.checkpoint.merak_incremental import _node_materialized_read_lock


def _hold_materialized_slot(
    run_id, lock_root, readers, started, entered, release, outcomes
):
    os.environ["CKPT_MATERIALIZED_LOCK_ROOT"] = lock_root
    os.environ["CKPT_MATERIALIZED_READERS_PER_NODE"] = str(readers)
    os.environ["CKPT_MATERIALIZED_READ_LOCK_TIMEOUT_SECONDS"] = "10"
    started.set()
    try:
        with _node_materialized_read_lock(run_id):
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release materialized reader")
        outcomes.put(None)
    except BaseException as error:
        outcomes.put(f"{type(error).__name__}: {error}")


def test_materialized_read_lock_admits_two_processes_and_blocks_third(tmp_path):
    context = multiprocessing.get_context("fork")
    outcomes = context.Queue()
    started = [context.Event() for _ in range(3)]
    entered = [context.Event() for _ in range(3)]
    release = [context.Event() for _ in range(3)]
    processes = [
        context.Process(
            target=_hold_materialized_slot,
            args=(
                "two-reader-test",
                str(tmp_path),
                2,
                started[index],
                entered[index],
                release[index],
                outcomes,
            ),
        )
        for index in range(3)
    ]

    try:
        processes[0].start()
        assert entered[0].wait(timeout=5)
        processes[1].start()
        assert entered[1].wait(timeout=5)
        processes[2].start()
        assert started[2].wait(timeout=5)
        assert not entered[2].wait(timeout=0.2)

        release[0].set()
        assert entered[2].wait(timeout=5)
        release[1].set()
        release[2].set()

        assert [outcomes.get(timeout=5) for _ in processes] == [None, None, None]
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
    finally:
        for event in release:
            event.set()
        for process in processes:
            if process.pid is None:
                continue
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.parametrize("readers", ["0", "3", "not-an-integer"])
def test_materialized_read_lock_rejects_unsupported_reader_count(
    monkeypatch, tmp_path, readers
):
    monkeypatch.setenv("CKPT_MATERIALIZED_LOCK_ROOT", str(tmp_path))
    monkeypatch.setenv("CKPT_MATERIALIZED_READERS_PER_NODE", readers)

    with pytest.raises(ValueError, match="materialized.*readers"):
        with _node_materialized_read_lock("invalid-reader-count"):
            pass
