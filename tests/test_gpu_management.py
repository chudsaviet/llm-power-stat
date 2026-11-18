"""Tests for the nvidia-smi wrappers: list_gpus, get_current_power,
set_power and restore_power, against the fake nvidia-smi."""

import pytest

import helpers

UUID_0 = "GPU-fake-uuid-0000"
UUID_1 = "GPU-fake-uuid-0001"


class TestListGpus:
    def test_parses_gpus(self, module, server, make_smi):
        make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            assert ps.gpus == [
                module.GPU("NVIDIA Fake GPU A", UUID_0, 300, 200),
                module.GPU("NVIDIA Fake GPU B", UUID_1, 250, 150),
            ]

    def test_sorted_by_index(self, module, server, make_smi):
        # The state file lists GPU 1 before GPU 0.
        gpus = helpers.default_gpus()
        gpus.reverse()
        make_smi(gpus)
        with helpers.make_power_stat(module, server=server) as ps:
            assert [gpu.uuid for gpu in ps.gpus] == [UUID_0, UUID_1]

    def test_original_power_limits_are_the_current_limits(self, module, server, make_smi):
        make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            # 280/200 W are the current limits, not the 300/250 W defaults.
            assert ps.original_power_limits == [280.0, 200.0]


class TestGetCurrentPower:
    def test_returns_draw_per_gpu_in_list_order(self, module, server, make_smi):
        make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            assert ps.get_current_power() == [100.0, 200.0]


class TestSetPower:
    def test_sets_each_gpu_in_order(self, module, server, make_smi):
        smi = make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            ps.set_power([250, 175])
            assert smi.sets == [(UUID_0, 250.0), (UUID_1, 175.0)]
            assert smi.limits() == {UUID_0: 250.0, UUID_1: 175.0}

    def test_noop_in_dry_run(self, module, server, make_smi):
        smi = make_smi()
        with helpers.make_power_stat(module, server=server, dry_run=True) as ps:
            ps.set_power([100, 100])
            assert smi.sets == []
            assert smi.limits() == {UUID_0: 280.0, UUID_1: 200.0}


class TestRestorePower:
    def test_restores_original_limits(self, module, server, make_smi):
        smi = make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            ps.set_power([200, 150])
            ps.restore_power()
            assert smi.sets == [
                (UUID_0, 200.0),
                (UUID_1, 150.0),
                (UUID_0, 280.0),
                (UUID_1, 200.0),
            ]
        # __exit__ restored the limits a second time; they are back to the originals.
        assert smi.limits() == {UUID_0: 280.0, UUID_1: 200.0}

    def test_keeps_going_when_one_gpu_fails(self, module, server, make_smi):
        smi = make_smi()
        smi.fail_pl([UUID_1])
        with pytest.raises(RuntimeError, match=UUID_1):
            with helpers.make_power_stat(module, server=server) as ps:
                ps.restore_power()
        # GPU 0 still got its limit back, even though GPU 1 failed.
        assert smi.limits()[UUID_0] == 280.0
        assert (UUID_0, 280.0) in smi.sets

    def test_noop_in_dry_run(self, module, server, make_smi):
        smi = make_smi()
        with helpers.make_power_stat(module, server=server, dry_run=True) as ps:
            ps.restore_power()
            assert smi.sets == []
