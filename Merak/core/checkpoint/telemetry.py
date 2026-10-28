"""Small, dependency-free helpers for checkpoint transport telemetry."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

IB_DATA_COUNTER_BYTES_PER_UNIT = 4
_COUNTER_PATTERN = re.compile(
    r"^(PortXmitData|PortRcvData|PortXmitPkts|PortRcvPkts):\.+([0-9]+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class IbPortCounters:
    xmit_data_units: int
    rcv_data_units: int
    xmit_packets: int
    rcv_packets: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class IbPortCounterDelta:
    xmit_bytes: int
    rcv_bytes: int
    xmit_packets: int
    rcv_packets: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def parse_perfquery_extended(output: str) -> IbPortCounters:
    """Parse the four 64-bit counters emitted by ``perfquery -x``."""

    fields = {name: int(value) for name, value in _COUNTER_PATTERN.findall(output)}
    required = {"PortXmitData", "PortRcvData", "PortXmitPkts", "PortRcvPkts"}
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"perfquery output is missing counters: {sorted(missing)}")
    return IbPortCounters(
        xmit_data_units=fields["PortXmitData"],
        rcv_data_units=fields["PortRcvData"],
        xmit_packets=fields["PortXmitPkts"],
        rcv_packets=fields["PortRcvPkts"],
    )


def _unsigned_delta(before: int, after: int, counter_bits: int) -> int:
    if before < 0 or after < 0:
        raise ValueError("counter values cannot be negative")
    modulus = 1 << counter_bits
    if before >= modulus or after >= modulus:
        raise ValueError(f"counter value does not fit in {counter_bits} bits")
    return after - before if after >= before else modulus - before + after


def ib_counter_delta(
    before: IbPortCounters,
    after: IbPortCounters,
    *,
    counter_bits: int = 64,
) -> IbPortCounterDelta:
    """Calculate a wrap-safe delta and convert IB data units to bytes."""

    if counter_bits < 1:
        raise ValueError("counter_bits must be positive")
    return IbPortCounterDelta(
        xmit_bytes=_unsigned_delta(
            before.xmit_data_units, after.xmit_data_units, counter_bits
        )
        * IB_DATA_COUNTER_BYTES_PER_UNIT,
        rcv_bytes=_unsigned_delta(
            before.rcv_data_units, after.rcv_data_units, counter_bits
        )
        * IB_DATA_COUNTER_BYTES_PER_UNIT,
        xmit_packets=_unsigned_delta(
            before.xmit_packets, after.xmit_packets, counter_bits
        ),
        rcv_packets=_unsigned_delta(
            before.rcv_packets, after.rcv_packets, counter_bits
        ),
    )
