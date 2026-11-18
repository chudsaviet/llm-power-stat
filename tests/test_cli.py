"""End-to-end tests: run llm-power-stat.py as a subprocess against the fakes."""

import csv
import signal
import subprocess
import sys
import time
from datetime import datetime

import pytest

import helpers

UUID_0 = "GPU-fake-uuid-0000"
UUID_1 = "GPU-fake-uuid-0001"
EXPECTED_LIMITS = [[300, 250], [200, 250], [300, 150], [200, 150]]
ORIGINAL_LIMITS = {UUID_0: 280.0, UUID_1: 200.0}


def base_command(server) -> list[str]:
    return [
        sys.executable,
        str(helpers.PROGRAM),
        "--api_endpoint",
        server.base_url,
        "--nvidia-smi-executable",
        str(helpers.FAKE_SMI),
        "--warmup_minutes",
        "0",
        "--power_step_W",
        "100",
    ]


class TestArgValidation:
    # Validation happens before any GPU or server access, so no fakes are needed.
    @pytest.mark.parametrize(
        ("extra_args", "message"),
        [
            (["--power_step_W", "0"], "must be positive"),
            (["--tokens_limit", "0"], "must be positive"),
            (["--warmup_minutes", "-1"], "must not be negative"),
            (["--sweeps", "0"], "must be positive"),
            (["--sweeps", "-2"], "must be positive"),
        ],
    )
    def test_rejects_invalid_arguments(self, extra_args, message):
        result = subprocess.run(
            [
                sys.executable,
                str(helpers.PROGRAM),
                "--api_endpoint",
                "localhost:1",
                *extra_args,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 2
        assert message in result.stderr


class TestDryRun:
    def test_rows_to_stdout_header_to_stderr_no_side_effects(self, server, make_smi, tmp_path):
        smi = make_smi()
        result = subprocess.run(
            base_command(server) + ["--dry_run"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        rows = [line for line in result.stdout.splitlines() if line]
        assert len(rows) == len(EXPECTED_LIMITS)
        assert "Timestamp" not in result.stdout
        for row, limits in zip(rows, EXPECTED_LIMITS):
            parts = row.split(",")
            datetime.fromisoformat(parts[0])
            assert len(parts) == 6
            assert parts[1:3] == [str(x) for x in limits]
            assert parts[3:5] == ["0", "0"]
            assert parts[5] == "0.0"
        assert (
            "Timestamp,GPU0 power limit W,GPU1 power limit W,"
            "GPU0 avg power W,GPU1 avg power W,TPS"
        ) in result.stderr
        # Dry run changes no power limits and sends no prompts.
        assert smi.sets == []
        assert server.requests == []


class TestSweepsOption:
    def test_sweeps_flag_repeats_full_sweep(self, server, make_smi):
        smi = make_smi()
        result = subprocess.run(
            base_command(server) + ["--dry_run", "--sweeps", "2"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        rows = [line for line in result.stdout.splitlines() if line]
        assert len(rows) == 2 * len(EXPECTED_LIMITS)
        assert "Sweep 2 of 2." in result.stderr
        assert smi.sets == []
        assert server.requests == []


class TestRealRun:
    def test_writes_csv_and_restores_original_limits(self, server, make_smi, tmp_path):
        smi = make_smi()
        out = tmp_path / "results.csv"
        result = subprocess.run(
            base_command(server) + ["--output_file", str(out)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0] == [
            "Timestamp",
            "GPU0 power limit W",
            "GPU1 power limit W",
            "GPU0 avg power W",
            "GPU1 avg power W",
            "TPS",
        ]
        assert len(rows) == 1 + len(EXPECTED_LIMITS)
        for row, limits in zip(rows[1:], EXPECTED_LIMITS):
            datetime.fromisoformat(row[0])
            assert row[1:3] == [str(x) for x in limits]
            assert row[3:5] == ["100", "200"]
            assert row[5] == "33.3"
        # The run ended with the pre-run (original) limits, not the defaults.
        assert smi.limits() == ORIGINAL_LIMITS
        assert smi.sets[-2:] == [(UUID_0, 280.0), (UUID_1, 200.0)]
        assert len(server.requests) == len(EXPECTED_LIMITS)


class TestSignals:
    @pytest.mark.parametrize(
        ("signum", "expected_code"),
        [(signal.SIGTERM, 143), (signal.SIGHUP, 129)],
    )
    def test_interrupted_run_restores_limits(self, server, make_smi, tmp_path, signum, expected_code):
        smi = make_smi()
        out = tmp_path / "results.csv"
        # Slow responses guarantee the process is mid-measurement when signaled.
        server.delay = 1.5
        proc = subprocess.Popen(
            base_command(server) + ["--output_file", str(out)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            # Wait for the first data row (rows are flushed right after being
            # written), then interrupt while the next measurement runs.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if out.exists() and len(out.read_text(encoding="utf-8").splitlines()) >= 2:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("the first data row was not written in time")
            proc.send_signal(signum)
            assert proc.wait(timeout=30) == expected_code
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        # The interrupted run still restored the original limits.
        assert smi.limits() == ORIGINAL_LIMITS
        assert smi.sets[-2:] == [(UUID_0, 280.0), (UUID_1, 200.0)]
