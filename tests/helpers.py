"""Shared helpers for the llm-power-stat test suite.

Provides the module loader for llm-power-stat.py, a threaded fake
llama.cpp server, and the state-file handle for the fake nvidia-smi
(tests/fake_nvidia_smi.py).
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
PROGRAM: Path = REPO_ROOT / "llm-power-stat.py"
FAKE_SMI: Path = Path(__file__).resolve().parent / "fake_nvidia_smi.py"

DEFAULT_PROMPT: str = "Count to ten."


def load_module():
    """Load llm-power-stat.py as a module.

    The file name contains hyphens, so a regular import is impossible
    (see AGENTS.md).
    """
    spec = importlib.util.spec_from_file_location("llm_power_stat", PROGRAM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_gpus() -> list[dict]:
    """Two fake GPUs.

    The current limits (280/200 W) deliberately differ from the defaults
    (300/250 W), so tests can tell "restored the originals" from
    "reset to the defaults".
    """
    return [
        {
            "index": 0,
            "uuid": "GPU-fake-uuid-0000",
            "model": "NVIDIA Fake GPU A",
            "default": 300,
            "min": 200,
            "limit": 280,
            "draw": 100.0,
        },
        {
            "index": 1,
            "uuid": "GPU-fake-uuid-0001",
            "model": "NVIDIA Fake GPU B",
            "default": 250,
            "min": 150,
            "limit": 200,
            "draw": 200.0,
        },
    ]


@dataclass
class FakeSMI:
    """A fake nvidia-smi state file plus accessors for the recorded calls."""

    path: Path
    gpus: list[dict] = field(default_factory=default_gpus)
    fail_pl_uuids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._write()

    def _write(self) -> None:
        state = {
            "gpus": self.gpus,
            "fail_pl_uuids": self.fail_pl_uuids,
            "log": [],
        }
        self.path.write_text(json.dumps(state), encoding="utf-8")

    def read_state(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def log(self) -> list[dict]:
        return self.read_state()["log"]

    # (uuid, watts) of every successful -pl, in call order.
    @property
    def sets(self) -> list[tuple[str, float]]:
        return [
            (entry["uuid"], float(entry["watts"]))
            for entry in self.log
            if entry["type"] == "set"
        ]

    def draws(self, uuid: str) -> list[float]:
        return [
            float(entry["watts"])
            for entry in self.log
            if entry["type"] == "draw" and entry["uuid"] == uuid
        ]

    def limits(self) -> dict[str, float]:
        return {
            gpu["uuid"]: float(gpu["limit"]) for gpu in self.read_state()["gpus"]
        }

    # Marks -pl calls for these GPUs to fail (non-zero exit, no state change).
    def fail_pl(self, uuids: list[str]) -> None:
        state = self.read_state()
        state["fail_pl_uuids"] = uuids
        self.path.write_text(json.dumps(state), encoding="utf-8")


class _SilentThreadingHTTPServer(ThreadingHTTPServer):
    # Abandoned in-flight requests (e.g. the client was killed) must not
    # dump tracebacks into the test output.
    def handle_error(self, request, client_address) -> None:
        pass


class FakeLLMCppServer:
    """Threaded stand-in for llama-server's /v1/chat/completions endpoint.

    Knobs (read per request, so they can change between calls):
      body    JSON response body, default {"timings": {"predicted_per_second": 33.3}}
      status  HTTP status code
      delay   seconds to sleep before responding
    Every request is appended to .requests as
      {"path": str, "headers": {lowercase-name: value}, "json": parsed body}.
    """

    def __init__(
        self,
        body: dict | None = None,
        status: int = 200,
        delay: float = 0.0,
    ) -> None:
        self.body: dict = (
            body if body is not None else {"timings": {"predicted_per_second": 33.3}}
        )
        self.status: int = status
        self.delay: float = delay
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b""
                try:
                    payload: Any = json.loads(raw) if raw else None
                except ValueError:
                    payload = None
                server.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "json": payload,
                    }
                )
                if server.delay > 0:
                    time.sleep(server.delay)
                response = json.dumps(server.body).encode("utf-8")
                self.send_response(server.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: Any) -> None:
                pass  # keep the test output clean

        self._server = _SilentThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port: int = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def make_power_stat(
    module,
    *,
    server: FakeLLMCppServer | None,
    dry_run: bool = False,
    api_key_file: str | None = None,
    power_step_W: int = 100,
    prompt: str = DEFAULT_PROMPT,
    tokens_limit: int = 42,
    warmup_minutes: int = 0,
):
    """Build an LLMPowerStat wired to the fakes.

    The fake nvidia-smi's state file must be selected via the
    FAKE_SMI_STATE environment variable (see the make_smi fixture).
    `server` may be None if the test never sends a prompt.
    """
    return module.LLMPowerStat(
        api_endpoint=server.base_url if server is not None else "127.0.0.1:1",
        api_key_file=api_key_file,
        nvidia_smi_executable=str(FAKE_SMI),
        power_step_W=power_step_W,
        prompt=prompt,
        tokens_limit=tokens_limit,
        warmup_minutes=warmup_minutes,
        dry_run=dry_run,
    )
