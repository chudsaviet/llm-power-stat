"""In-process tests for run(): the full measurement sweep and CSV output."""

import csv
from datetime import datetime

import pytest

import helpers

UUID_0 = "GPU-fake-uuid-0000"
UUID_1 = "GPU-fake-uuid-0001"
# gen_power_steps order for the default fake GPUs with power_step_W=100.
EXPECTED_LIMITS = [[300, 250], [200, 250], [300, 150], [200, 150]]


@pytest.fixture(autouse=True)
def fast_sampling(module, monkeypatch):
    monkeypatch.setattr(module, "POWER_SAMPLING_INTERVAL_SEC", 0.05)


def read_rows(path) -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


def check_data_rows(rows: list[list[str]], module) -> None:
    assert rows[0] == module.csv_header(2)
    assert len(rows) == 1 + len(EXPECTED_LIMITS)
    for row, limits in zip(rows[1:], EXPECTED_LIMITS):
        datetime.fromisoformat(row[0])  # raises on a bad timestamp
        assert row[1:3] == [str(x) for x in limits]
        assert row[3:5] == ["100", "200"]
        assert row[5] == "33.3"


class TestRun:
    def test_writes_one_row_per_power_combination(self, module, server, make_smi, tmp_path):
        smi = make_smi()
        out = tmp_path / "results.csv"
        with helpers.make_power_stat(module, server=server, power_step_W=100) as ps:
            module.run(ps, str(out))
        check_data_rows(read_rows(out), module)
        # One prompt per combination, and the exit restored the original limits.
        assert len(server.requests) == len(EXPECTED_LIMITS)
        assert smi.sets[-2:] == [(UUID_0, 280.0), (UUID_1, 200.0)]
        assert smi.limits() == {UUID_0: 280.0, UUID_1: 200.0}

    def test_appends_to_existing_file_without_repeating_header(self, module, server, make_smi, tmp_path):
        out = tmp_path / "results.csv"
        for _ in range(2):
            make_smi()  # fresh state file per run
            with helpers.make_power_stat(module, server=server, power_step_W=100) as ps:
                module.run(ps, str(out))
        rows = read_rows(out)
        assert rows.count(module.csv_header(2)) == 1
        assert len(rows) == 1 + 2 * len(EXPECTED_LIMITS)

    def test_dry_run_writes_zero_rows_without_side_effects(self, module, server, make_smi, tmp_path):
        smi = make_smi()
        out = tmp_path / "results.csv"
        with helpers.make_power_stat(
            module, server=server, power_step_W=100, dry_run=True
        ) as ps:
            module.run(ps, str(out))
        rows = read_rows(out)
        assert rows[0] == module.csv_header(2)
        for row, limits in zip(rows[1:], EXPECTED_LIMITS):
            # The limit columns still contain the swept values;
            # the measurements are zeros.
            assert row[1:3] == [str(x) for x in limits]
            assert row[3:5] == ["0", "0"]
            assert row[5] == "0.0"
        # No power limits were changed and no prompts were sent.
        assert smi.sets == []
        assert server.requests == []

    def test_warmup_sets_default_limits_and_sends_no_prompts(self, module, server, make_smi):
        smi = make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            # 0 minutes: the warmup loop body never runs.
            ps.warmup()
            assert smi.sets == [(UUID_0, 300.0), (UUID_1, 250.0)]
            assert server.requests == []
