# llm-power-stat

Measures how GPU power limits affect LLM inference speed.

For every combination of per-GPU power limits, the script:

1. sets the limits with `nvidia-smi -pl`,
2. sends one prompt to a running **llama.cpp** server (`llama-server`),
3. samples the real power draw of every GPU about once per second while the prompt runs,
4. writes one CSV row: timestamp, power limits, average power draw, tokens per second (TPS).

The power limits are swept **downward**: each GPU goes from its default limit to its minimum limit in steps of `--power_step_W` (the default limit is always tested, and the script never goes above it). GPU 0 changes fastest, so every combination is covered.

At the end of the run — and on any error, Ctrl+C, SIGTERM or SIGHUP — every GPU's power limit is restored to the value it had before the script started.

## Output

CSV, comma-separated, one row per measurement. Column order for N GPUs:

```
Timestamp, GPU0 power limit W ... GPU(N-1) power limit W, GPU0 avg power W ... GPU(N-1) avg power W, TPS
```

Example with 2 GPUs:

```csv
Timestamp,GPU0 power limit W,GPU1 power limit W,GPU0 avg power W,GPU1 avg power W,TPS
2026-11-17T14:48:23-08:00,300,250,287,241,33.3
2026-11-17T14:48:41-08:00,200,250,195,238,31.1
```

- `Timestamp` is ISO 8601 in local time with UTC offset, taken when the prompt finished.
- `TPS` comes from llama.cpp's own measurement (`timings.predicted_per_second`), excluding prompt processing. llama.cpp is the only supported server.
- With `--output_file` the file is **appended**; the header is written only if the file is new or empty.
- Without `--output_file`, data rows go to STDOUT (safe to pipe) and all diagnostics, including the header, go to STDERR.

## Requirements

- Linux with NVIDIA GPUs and `nvidia-smi`
- Python **3.11 or newer**
- A running llama.cpp server (`llama-server`)
- Root (or equivalent permission) — setting power limits with `nvidia-smi -pl` usually needs it

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Safe check: changes no power limits and sends no prompts.
.venv/bin/python llm-power-stat.py --api_endpoint localhost:8080 --warmup_minutes 0 --dry_run

# Real run, writing to a CSV file.
sudo .venv/bin/python llm-power-stat.py --api_endpoint localhost:8080 --output_file results.csv
```

`--api_endpoint` accepts `localhost:8080`, `http://host:8080` and OpenAI-style `http://host:8080/v1` (a trailing `/v1` is dropped; `http://` is assumed if no scheme is given).

### Options

| Option | Default | Meaning |
|---|---|---|
| `--api_endpoint` | required | llama.cpp server address. |
| `--output_file` | none | CSV file. Rows are appended. Without it, rows go to STDOUT and the header to STDERR. |
| `--api_key_file` | none | File containing the API key, sent as `Authorization: Bearer <key>`. |
| `--nvidia-smi-executable` | `/usr/bin/nvidia-smi` | Path to `nvidia-smi`. |
| `--power_step_W` | `10` | Watts between tested power limits. Must be > 0. |
| `--prompt` | Pi calculation prompt | Prompt sent for every measurement. |
| `--tokens_limit` | `3600` | `max_tokens` for each prompt. Must be > 0. |
| `--warmup_minutes` | `10` | Minutes of prompts at default power before measuring. Must be >= 0. |
| `--sweeps` | `1` | Number of times the full power sweep is repeated. Must be > 0. |
| `--dry_run` | off | Do not change power limits and do not send prompts. Rows contain zeros. |

## Notes

- The script restores the original power limits on every normal or interrupted exit. The only cases where restore is impossible are `kill -9`, a Python crash and power loss.
- Warmup runs at the GPUs' default power limits before the sweep starts.
- The API key file is read once at startup and is never printed, including in error messages.

## Testing

The test suite runs the whole program against a fake `nvidia-smi` and a fake llama.cpp server — no GPUs, no server and no root needed:

```bash
.venv/bin/python -m pytest tests/ -v
```

## License

[MIT](LICENSE) © 2026 Timofei Korostelev
