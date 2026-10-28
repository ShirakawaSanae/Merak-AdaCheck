import pytest

from Merak.core.mpu.initialize import mesh_layout
from Merak.core.mpu.topology import ProcessTopology


def test_data_first_mesh_places_dp_replicas_on_distinct_hosts():
    shape, names = mesh_layout(4, 2, 1, 2, "dp,pp,sp,tp")
    topology = ProcessTopology(list(names), list(shape))

    assert shape == (2, 4, 1, 2)
    for pipeline in range(4):
        for tensor in range(2):
            first = topology.get_rank(dp=0, pp=pipeline, sp=0, tp=tensor)
            second = topology.get_rank(dp=1, pp=pipeline, sp=0, tp=tensor)
            assert first < 8 <= second


def test_mesh_order_must_be_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        mesh_layout(4, 2, 1, 2, "pp,pp,sp,tp")
