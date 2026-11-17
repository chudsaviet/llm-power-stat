# AGENTS.md

Guide for AI coding agents working on this repository. Read it fully before changing code.

## What this project does

`llm-power-stat.py` measures how the GPU power limit affects LLM inference speed.

For every combination of GPU power limits, it:

1. Sets the power limits with `nvidia-smi -pl`.
2. Sends one prompt to a **llama.cpp server**.
3. While the prompt runs, reads the real power draw of every GPU about once per second.
4. Writes one CSV row: timestamp, power limits, average power draw, tokens per second (TPS).

At the end, or on any error, it puts every GPU's power limit back to what it was before the script started.

## Files

| File | Purpose |
|---|---|
| `llm-power-stat.py` | The whole program. There are no other source files. |
| `requirements.txt` | Runtime dependencies (only `httpx`). |
| `LICENSE` | License. |

There are no tests in the repository right now.

## Requirements

- Linux with NVIDIA GPUs and `nvidia-smi`.
- Python **3.11 or newer** (the code uses `typing.Self` and `asyncio.Runner`). The local `.venv` uses Python 3.14.
- A running llama.cpp server (`llama-server`).
- Setting power limits with `nvidia-smi -pl` usually needs root.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Safe check: changes no power limits and sends no prompts.
.venv/bin/python llm-power-stat.py --api_endpoint localhost:8080 --warmup_minutes 0 --dry_run

# Real run, writing to a CSV file.
sudo .venv/bin/python llm-power-stat.py --api_endpoint localhost:8080 --output_file results.csv
```

Command-line options:

| Option | Default | Meaning |
|---|---|---|
| `--api_endpoint` | required | llama.cpp server address. `localhost:8080`, `http://host:8080` and `http://host:8080/v1` all work. |
| `--output_file` | none | CSV file. Rows are appended. The header is written only if the file is new or empty. Without this option, rows go to STDOUT and the header goes to STDERR. |
| `--api_key_file` | none | File containing the API key, sent as `Authorization: Bearer <key>`. |
| `--nvidia-smi-executable` | `/usr/bin/nvidia-smi` | Path to `nvidia-smi`. Note: this option uses hyphens, all others use underscores. |
| `--power_step_W` | `10` | Watts between tested power limits. Must be > 0. |
| `--prompt` | Pi calculation prompt | Prompt sent for every measurement. |
| `--tokens_limit` | `3600` | `max_tokens` for each prompt. Must be > 0. |
| `--warmup_minutes` | `10` | Minutes of prompts at default power before measuring. Must be >= 0. |
| `--dry_run` | off | Do not change power limits and do not send prompts. Rows contain zeros. |

## Output format

CSV, comma-separated, no spaces, `\n` line endings, written with Python's `csv` module.
Column order for N GPUs:

```
Timestamp, GPU0 power limit W ... GPU(N-1) power limit W, GPU0 avg power W ... GPU(N-1) avg power W, TPS
```

Example with 2 GPUs:

```
Timestamp,GPU0 power limit W,GPU1 power limit W,GPU0 avg power W,GPU1 avg power W,TPS
2026-11-17T14:48:23-08:00,300,300,150,150,33.3
```

- `Timestamp` is ISO 8601 in local time with UTC offset, taken when the prompt finished.
- If you change the row layout in `run_analysis()`, change `csv_header()` the same way. They must always match.

## How the code is organized

Everything is in `llm-power-stat.py`, in this order:

1. **Module helpers**
   - `log()`: prints to STDERR. Ignores write errors on purpose (see "Safety rules").
   - `raise_system_exit()`: signal handler that turns SIGTERM/SIGHUP into `SystemExit`.
   - `to_base_url()`: normalizes `--api_endpoint` (adds `http://`, removes a trailing `/v1`).
2. **`GPU`**: a `NamedTuple` with `model_name`, `uuid`, `power_default_limit`, `power_min_limit` (whole watts).
3. **`LLMPowerStat`**: the main class. It is a **context manager** (`with LLMPowerStat(...) as s:`).
   - `__init__`: lists GPUs, saves each GPU's current limit in `original_power_limits`, reads the API key, creates the event loop (`asyncio.Runner`) and the `httpx.AsyncClient`.
   - `__exit__`: calls `restore_power()`, then closes the HTTP client and the event loop.
   - `run_smi()`, `query_gpu()`, `list_gpus()`, `get_current_power()`: wrappers around `nvidia-smi`.
   - `set_power()`: sets limits; stops at the first error. Does nothing in dry run.
   - `restore_power()`: sets the saved limits back; continues past failing GPUs, then raises. Does nothing in dry run.
   - `run_prompt()`: **synchronous** entry point. Runs `_run_prompt_async()` on the object's event loop.
   - `_run_prompt_async()`: starts the HTTP POST as an asyncio task, samples power every `POWER_SAMPLING_INTERVAL_SEC` until the task finishes, returns `(tps, [average watts per GPU])`.
   - `parse_tps()`: reads `timings.predicted_per_second` from the llama.cpp response.
   - `warmup()`, `gen_power_steps()`, `run_analysis()`: the measurement sweep.
4. **`main()`**: parses and validates arguments, installs signal handlers, creates `LLMPowerStat` in a `with` block, calls `run()`.
5. **`csv_header()`** and **`run()`**: open the output, run warmup and the sweep, write rows.

### Power steps

`gen_power_steps()` yields every combination of limits. Each GPU goes **down** from its default limit to its minimum limit in steps of `--power_step_W`. The default limit is always the first value. GPU 0 changes fastest. Example (GPU A: 300→200, GPU B: 250→150, step 100):

```
[300, 250], [200, 250], [300, 150], [200, 150]
```

### Async design (important)

Most functions are normal synchronous functions. Only the prompt request is async, so that power can be sampled while the request runs.

- There is **one** `asyncio.Runner` per `LLMPowerStat` object (`self.runner`). Run async code with `self.runner.run(...)`.
- **Do not** use `asyncio.run()`. It creates a new event loop each time, and the `httpx.AsyncClient` breaks when used from a different loop.
- **Do not** call `asyncio.create_task()` or `await` in synchronous functions. They only work inside `async def` functions that run on the loop.
- `nvidia-smi` calls block. Inside async code, run them with `await asyncio.to_thread(...)`.
- If power sampling fails, the request task is cancelled in a `finally` block. Keep that, otherwise the request keeps running on the shared loop.

## Safety rules (do not break these)

The script changes real hardware settings. A bug can leave GPUs power-limited after the script ends.

1. **Never set a limit above `power_default_limit`.** The script does not use the GPU's maximum limit.
2. **Always restore the original limits.** All work that changes limits must happen inside the `with LLMPowerStat(...)` block, so `__exit__` runs on normal exit, exceptions, Ctrl+C, SIGTERM and SIGHUP.
3. **`restore_power()` must try every GPU**, even if one fails.
4. **`log()` must never raise.** After an SSH disconnect, writing to STDERR fails. If `log()` raised, the restore would not run.
5. **Dry run must not change limits and must not send prompts.** `set_power()` and `restore_power()` return early when `self.dry_run` is true.
6. Restore is impossible after `kill -9`, a Python crash or a power loss. This is a known limit.

## Conventions

- **STDOUT is only for CSV rows.** All other output goes through `log()` (STDERR).
- Use type hints on variables and functions, like the existing code.
- Keep the program in the single file `llm-power-stat.py`.
- Only llama.cpp is supported. TPS comes from llama.cpp's `timings.predicted_per_second`. Do not add a fallback for other servers.
- The API key must never appear in error messages. That is why `httpx.LocalProtocolError` is caught and replaced, and why the key is checked for bad characters in `__init__`.
- `nvidia-smi` is called without `--format`. Output is parsed as: line 0 = header, line 1 = values, comma-separated, with a ` W` suffix to strip. Keep this format if you add queries.

## Testing changes without GPUs

There is no test suite. To check a change without real GPUs or a server:

1. Use `--dry_run` to check argument handling and CSV output.
2. For anything else, write a fake `nvidia-smi` script (pass it with `--nvidia-smi-executable`) that prints output in the same format, and a small local HTTP server that answers `POST /v1/chat/completions` with `{"timings": {"predicted_per_second": 33.3}}`.

Put these helper files outside the repository unless the maintainer asks for tests.

Because the file name contains hyphens, `import llm-power-stat` does not work. Load it like this:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("llm_power_stat", "llm-power-stat.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

## Working with the maintainer

- **Do not edit code unless the maintainer explicitly asks for a change.** For questions or reviews, answer and stop.
- Commit directly to `main`. Do not create feature branches.
- Commit or push only when asked.
