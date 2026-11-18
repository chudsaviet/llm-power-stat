"""Shared fixtures for the test suite."""

import os

import pytest

import helpers


@pytest.fixture(scope="session", autouse=True)
def executable_fake_smi() -> None:
    # Guard against checkouts that lost the executable bit.
    os.chmod(helpers.FAKE_SMI, 0o755)


@pytest.fixture(scope="session")
def module():
    return helpers.load_module()


@pytest.fixture()
def server():
    srv = helpers.FakeLLMCppServer()
    yield srv
    srv.stop()


@pytest.fixture()
def make_smi(tmp_path, monkeypatch):
    """Create a fake nvidia-smi state file and point FAKE_SMI_STATE at it."""

    def _make(gpus: list[dict] | None = None) -> helpers.FakeSMI:
        fake = helpers.FakeSMI(
            path=tmp_path / "smi_state.json",
            gpus=gpus if gpus is not None else helpers.default_gpus(),
        )
        monkeypatch.setenv("FAKE_SMI_STATE", str(fake.path))
        return fake

    return _make
