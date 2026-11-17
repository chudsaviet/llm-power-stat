import argparse
import asyncio
import re
import subprocess
import textwrap
import time
from collections.abc import Iterator
from typing import NamedTuple, Self

import httpx

SMI_EXECUTABLE: str = "/usr/bin/nvidia-smi"
POWER_SAMPLING_INTERVAL_SEC: float = 1.0

# TODO: Remember power limits before changing them, and restore them on exit even in case of error!


class GPU(NamedTuple):
    """Installed system GPU description."""

    model_name: str
    uuid: str
    # The script never goes to power_max_limit in order to keep safe operation.
    power_default_limit: int
    power_min_limit: int


# Matches lines like: "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-1234abcd-...)"
GPU_LINE_RE = re.compile(
    r"^GPU\s+(?P<index>\d+):\s+(?P<model_name>.+?)\s+\(UUID:\s*(?P<uuid>[^)\s]+)\s*\)\s*$"
)


class LLMPowerStat:
    """Measures GPU power settings vs LLM inference performance."""

    def __init__(
        self,
        api_endpoint: str,
        api_key_file: str | None,
        nvidia_smi_executable: str,
        power_step_W: int,
        prompt: str,
        tokens_limit: int,
        warmup_minutes: int,
        dry_run: bool,
    ) -> None:
        self.api_endpoint: str = api_endpoint
        self.api_key_file: str | None = api_key_file
        self.nvidia_smi_executable: str = nvidia_smi_executable
        self.power_step_W: int = power_step_W
        self.prompt: str = prompt
        self.tokens_limit: int = tokens_limit
        self.warmup_minutes: int = warmup_minutes
        self.dry_run = dry_run

        print("Listing local GPUs...")
        self.gpus: list[GPU] = self.list_gpus()

        self.api_key: str | None = None
        if self.api_key_file is not None:
            with open(self.api_key_file) as f:
                # Strip the trailing newline: h11 rejects header values containing "\n".
                self.api_key = f.read().strip()
            # Fail early with our own message: h11's header error would print the key.
            if not (self.api_key.isascii() and self.api_key.isprintable()):
                raise ValueError(
                    f"API key in {self.api_key_file} contains non-printable or non-ASCII characters."
                )

        headers: dict[str, str] = {}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # One event loop for the object's lifetime: the AsyncClient's connection pool
        # is bound to the loop it first runs on, so asyncio.run() per call would break it.
        self.runner: asyncio.Runner = asyncio.Runner()
        self.http_client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=f"http://{self.api_endpoint}",
            headers=headers,
            timeout=httpx.Timeout(10.0, read=None),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.runner.run(self.http_client.aclose())
        finally:
            self.runner.close()

    # Will throw exception if anything wrong.
    # Returns the stdout. Ignores the stderr.
    def run_smi(self, parameters: list[str]) -> str:
        return subprocess.run(
            [self.nvidia_smi_executable] + parameters,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout

    def query_gpu(self, uuid: str, columns: list[str]) -> list[str]:
        text = self.run_smi(["-i", uuid, "--query-gpu", ",".join(columns)])
        return [x.strip() for x in text.split("\n")[1].split(",")]

    # Returns GPUs sorted by nvidia-smi GPU index (GPU 0, GPU 1, ...).
    # The index itself is not stored in GPU; list position is the only record of it.
    def list_gpus(self) -> list[GPU]:
        text = self.run_smi(["--list-gpus"])
        indexed: list[tuple[int, str, str]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            match = GPU_LINE_RE.match(line)
            if match is None:
                raise ValueError(f"Unexpected nvidia-smi --list-gpus line: {line!r}")
            indexed.append((int(match["index"]), match["model_name"], match["uuid"]))
        indexed.sort(key=lambda item: item[0])

        gpus: list[GPU] = []
        for _, model_name, uuid in indexed:
            gpu_parameters = self.query_gpu(
                uuid,
                [
                    "power.default_limit",
                    "power.min_limit",
                ],
            )
            gpus.append(
                GPU(
                    model_name=model_name,
                    uuid=uuid,
                    power_default_limit=int(
                        float(gpu_parameters[0].rstrip("W").strip())
                    ),
                    power_min_limit=int(float(gpu_parameters[1].rstrip("W").strip())),
                )
            )

        return gpus

    # Prints GPUs in list order, numbered the same way nvidia-smi does: "GPU 0: ...".
    def print_gpus(self) -> None:
        for index, gpu in enumerate(self.gpus):
            print(f"GPU {index}: {gpu.model_name} (UUID: {gpu.uuid})")
            print(f"  Power default limit: {gpu.power_default_limit}W")
            print(f"  Power min limit: {gpu.power_min_limit}W")

    def get_current_power(self) -> list[float]:
        result: list[float] = []
        for gpu in self.gpus:
            smi_output = self.run_smi(["-i", gpu.uuid, "--query-gpu", "power.draw"])
            result.append(float(smi_output.split("\n")[1].rstrip("W").strip()))
        return result

    # Sync entry point: runs the async version on the object's event loop.
    def run_prompt(self) -> tuple[float, list[int]]:
        return self.runner.run(self._run_prompt_async())

    async def _run_prompt_async(self) -> tuple[float, list[int]]:
        power_checks: int = 0
        power_accumulators: list[float] = [0.0 for x in self.gpus]
        try:
            async_task: asyncio.Task = asyncio.create_task(
                self.http_client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": self.prompt}],
                        "max_tokens": self.tokens_limit,
                    },
                )
            )

            response: httpx.Response | None = None
            try:
                while True:
                    # TODO: We will have average power = 0.0 if request finished faster than POWER_SAMPLING_INTERVAL_SEC.
                    # We need a fix for this case.
                    done, _ = await asyncio.wait(
                        [async_task], timeout=POWER_SAMPLING_INTERVAL_SEC
                    )
                    if done:
                        response = async_task.result()
                        break

                    power_checks += 1
                    # nvidia-smi blocks, so run it in a thread to keep the POST progressing.
                    current_power: list[float] = await asyncio.to_thread(
                        self.get_current_power
                    )
                    for i in range(len(self.gpus)):
                        power_accumulators[i] += current_power[i]
            finally:
                # If sampling failed, don't leave the request running on the loop.
                async_task.cancel()

        except httpx.LocalProtocolError:
            # h11's message includes the offending header value, i.e. the API key.
            # "from None" keeps Python from printing the original exception too.
            raise RuntimeError(
                "Invalid HTTP request headers (check the API key file)."
            ) from None

        if response.status_code != 200:
            raise SystemError(f"API endpoint returned {response.status_code}!")

        return float(response.json()["timings"]["predicted_per_second"]), [
            int(x / power_checks) for x in power_accumulators
        ]

    # It can warm up for a little more minutes than specified, because it just runs the query until the time is up.
    # It will not kill the last query.
    def warmup(self) -> None:
        print(f"Warming up for at least {self.warmup_minutes} minutes...")
        print("Setting default power limits...")
        self.set_power([x.power_default_limit for x in self.gpus])
        print("Running prompts...")

        start_time: int = int(time.monotonic())
        if not self.dry_run:
            while int(time.monotonic()) < start_time + self.warmup_minutes * 60:
                print(".", end="", flush=True)
                self.run_prompt()
        print(flush=True)

        print(
            f"Warmed up for {int((int(time.monotonic()) - start_time) / 60)} minutes."
        )

    # No-op in dry run, so callers don't need to check.
    def set_power(self, power_values: list[int]):
        if self.dry_run:
            return
        for i, gpu in enumerate(self.gpus):
            self.run_smi(["-i", gpu.uuid, "-pl", str(power_values[i])])

    def gen_power_steps(self, gpus: list[GPU]) -> Iterator[list[int]]:
        if len(gpus) > 0:
            gpu: GPU = gpus.pop()
            for power in reversed(
                range(
                    gpu.power_min_limit,
                    gpu.power_default_limit + self.power_step_W,
                    self.power_step_W,
                )
            ):
                if power > gpu.power_default_limit:
                    continue

                if len(gpus) > 0:
                    for head in self.gen_power_steps(gpus.copy()):
                        head.append(power)
                        yield head
                else:
                    yield [power]

    def run_analysis(self) -> Iterator[list]:
        power_values: Iterator[list[int]] = self.gen_power_steps(self.gpus.copy())

        for power_value in power_values:
            print(
                f"Setting power limits to {power_value}. Running prompt...",
                end="",
                flush=True,
            )
            self.set_power(power_value)
            tps, power_averages = (
                self.run_prompt() if not self.dry_run else (0.0, [0 for x in self.gpus])
            )

            print(f" TPS={tps:.2f} Power averages={power_averages}", flush=True)
            yield power_value + [str(x) for x in power_averages] + [tps]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="LLM power stat",
        description="A set of scripts to measure GPU power settings vs LLM inference performance.",
    )

    parser.add_argument(
        "--api_endpoint",
        type=str,
        required=True,
        help="OpenAI-compatible API endpoint.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help=textwrap.dedent("""Path to the output CSV file. 
                             It will be created if not exists. 
                             If it exists, rows will simply be added to the bottom.
                             If not set - header will be printed to STDERR, data rows will be printed to STDOUT."""),
    )

    parser.add_argument(
        "--api_key_file",
        type=str,
        required=False,
        help="Path to a file containing API key.",
    )

    parser.add_argument(
        "--nvidia-smi-executable",
        type=str,
        required=False,
        help="Path to the nvidia-smi executable.",
        default=SMI_EXECUTABLE,
    )

    parser.add_argument(
        "--power_step_W",
        type=int,
        required=False,
        help="Size of each power step in Watts.",
        default=10,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="Promt to submit to LLM.",
        default="Run full calculation of Pi using the Chudnovsky series down to 128th digit.",
    )

    parser.add_argument(
        "--tokens_limit",
        type=int,
        required=False,
        help="Max number of tokens to be generated on each test.",
        default=3600,
    )

    parser.add_argument(
        "--warmup_minutes",
        type=int,
        required=False,
        help="Number of minutes to run a warmup queries before starting tests.",
        default=10,
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run.",
    )

    args = parser.parse_args()

    print(
        f"Running power vs tokens-per-second test using http://{args.api_endpoint} API endpoint."
    )

    with LLMPowerStat(
        api_endpoint=args.api_endpoint,
        api_key_file=args.api_key_file,
        nvidia_smi_executable=args.nvidia_smi_executable,
        power_step_W=args.power_step_W,
        prompt=args.prompt,
        tokens_limit=args.tokens_limit,
        warmup_minutes=args.warmup_minutes,
        dry_run=args.dry_run,
    ) as power_stat:
        run(power_stat, args.output_file)


def run(power_stat: LLMPowerStat, output_file: str | None) -> None:
    power_stat.print_gpus()
    if power_stat.warmup_minutes > 0:
        power_stat.warmup()

    results: Iterator = power_stat.run_analysis()

    if output_file is None:
        for row in results:
            print(", ".join([str(x) for x in row]))
        return

    with open(output_file, "+w") as f:
        f.write(
            f"{', '.join([f'GPU{x!s} watts' for x in range(len(power_stat.gpus))])}, TPS\n"
        )
        for row in results:
            f.write(", ".join([str(x) for x in row]))
            f.write("\n")
            f.flush()


if __name__ == "__main__":
    main()
