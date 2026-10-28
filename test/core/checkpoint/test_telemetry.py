import pytest

from Merak.core.checkpoint.telemetry import (
    IB_DATA_COUNTER_BYTES_PER_UNIT,
    ib_counter_delta,
    parse_perfquery_extended,
)


PERFQUERY_OUTPUT = """
# Port extended counters: Lid 220 port 1
PortSelect:......................1
PortXmitData:....................19899146
PortRcvData:.....................1690309814
PortXmitPkts:....................46792
PortRcvPkts:.....................5990640
"""


def test_parse_and_convert_perfquery_counters():
    before = parse_perfquery_extended(PERFQUERY_OUTPUT)
    after = type(before)(
        xmit_data_units=before.xmit_data_units + 25,
        rcv_data_units=before.rcv_data_units + 50,
        xmit_packets=before.xmit_packets + 2,
        rcv_packets=before.rcv_packets + 3,
    )

    delta = ib_counter_delta(before, after)

    assert delta.xmit_bytes == 25 * IB_DATA_COUNTER_BYTES_PER_UNIT
    assert delta.rcv_bytes == 50 * IB_DATA_COUNTER_BYTES_PER_UNIT
    assert delta.xmit_packets == 2
    assert delta.rcv_packets == 3


def test_counter_delta_handles_wrap():
    before = parse_perfquery_extended(PERFQUERY_OUTPUT)
    wrapped_before = type(before)(2**64 - 2, 2**64 - 3, 2**64 - 4, 2**64 - 5)
    wrapped_after = type(before)(3, 4, 5, 6)

    delta = ib_counter_delta(wrapped_before, wrapped_after)

    assert delta.xmit_bytes == 5 * IB_DATA_COUNTER_BYTES_PER_UNIT
    assert delta.rcv_bytes == 7 * IB_DATA_COUNTER_BYTES_PER_UNIT
    assert delta.xmit_packets == 9
    assert delta.rcv_packets == 11


def test_perfquery_parser_rejects_partial_output():
    with pytest.raises(ValueError, match="missing"):
        parse_perfquery_extended("PortXmitData:....................1\n")
