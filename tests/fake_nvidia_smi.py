#!/usr/bin/env python3
"""Fake nvidia-smi for the test suite.

Supports the three invocation forms llm-power-stat.py uses:

    --list-gpus
    -i <uuid> --query-gpu <column[,column...]>
    -i <uuid> -pl <watts>

State lives in the JSON file named by the FAKE_SMI_STATE environment
variable. The schema is defined in tests/helpers.py.
"""

import json
import os
import sys

# Maps --query-gpu column names to the keys used in the state file.
COLUMN_KEYS: dict[str, str] = {
    "power.default_limit": "default",
    "power.min_limit": "min",
    "power.limit": "limit",
}


def fail(message: str) -> None:
    print(f"fake nvidia-smi: {message}", file=sys.stderr)
    sys.exit(1)


def format_watts(value: float) -> str:
    return f"{value:g} W"


def write_state(state: dict, state_path: str) -> None:
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def main() -> None:
    args: list[str] = sys.argv[1:]
    state_path: str | None = os.environ.get("FAKE_SMI_STATE")
    if state_path is None:
        fail("FAKE_SMI_STATE is not set")
    with open(state_path, encoding="utf-8") as f:
        state: dict = json.load(f)
    gpus: list[dict] = state["gpus"]

    if args == ["--list-gpus"]:
        for gpu in gpus:
            print(f"GPU {gpu['index']}: {gpu['model']} (UUID: {gpu['uuid']})")
        return

    if "-i" not in args:
        fail(f"unsupported arguments: {args!r}")
    uuid: str = args[args.index("-i") + 1]
    gpu: dict | None = next((g for g in gpus if g["uuid"] == uuid), None)
    if gpu is None:
        fail(f"unknown GPU UUID: {uuid}")

    if "--query-gpu" in args:
        columns: list[str] = args[args.index("--query-gpu") + 1].split(",")
        values: list[str] = []
        for column in columns:
            if column in COLUMN_KEYS:
                values.append(format_watts(float(gpu[COLUMN_KEYS[column]])))
            elif column == "power.draw":
                # Optional counting mode: each served draw is base + step * calls,
                # so tests can distinguish "average of samples" from "last sample".
                draw: float = float(
                    gpu["draw"] + gpu.get("draw_step", 0) * gpu.get("draw_count", 0)
                )
                gpu["draw_count"] = gpu.get("draw_count", 0) + 1
                state["log"].append({"type": "draw", "uuid": uuid, "watts": draw})
                values.append(format_watts(draw))
            else:
                fail(f"unknown column: {column}")
        print(",".join(columns))
        print(", ".join(values))
        write_state(state, state_path)
        return

    if "-pl" in args:
        watts: float = float(args[args.index("-pl") + 1])
        if uuid in state.get("fail_pl_uuids", []):
            fail(f"could not set power limit on GPU {uuid} (injected failure)")
        gpu["limit"] = watts
        state["log"].append({"type": "set", "uuid": uuid, "watts": watts})
        write_state(state, state_path)
        print(f"Applied new power limit to GPU {uuid}.")
        return

    fail(f"unsupported arguments: {args!r}")


if __name__ == "__main__":
    main()
