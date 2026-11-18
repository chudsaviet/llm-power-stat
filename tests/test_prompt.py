"""Tests for run_prompt() against the fake nvidia-smi and the fake llama.cpp server."""

import pytest

import helpers


@pytest.fixture(autouse=True)
def fast_sampling(module, monkeypatch):
    # The real 1s sampling interval would make the suite slow; the value is a
    # module global read at call time, so patching it is safe.
    monkeypatch.setattr(module, "POWER_SAMPLING_INTERVAL_SEC", 0.05)


class TestRunPrompt:
    def test_returns_tps_and_average_power(self, module, server, make_smi):
        make_smi()
        with helpers.make_power_stat(
            module, server=server, prompt="Count to ten.", tokens_limit=42
        ) as ps:
            tps, averages = ps.run_prompt()
        assert tps == 33.3
        assert averages == [100, 200]

    def test_sends_prompt_and_tokens_limit(self, module, server, make_smi):
        make_smi()
        with helpers.make_power_stat(
            module, server=server, prompt="Count to ten.", tokens_limit=42
        ) as ps:
            ps.run_prompt()
        request = server.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["json"]["messages"] == [{"role": "user", "content": "Count to ten."}]
        assert request["json"]["max_tokens"] == 42

    def test_sends_deterministic_sampling_settings(self, module, server, make_smi):
        make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            ps.run_prompt()
        assert server.requests[0]["json"]["temperature"] == 0.0
        assert server.requests[0]["json"]["top_k"] == 1

    def test_seed_sent_when_given(self, module, server, make_smi):
        make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            ps.run_prompt(seed=12345)
        assert server.requests[0]["json"]["seed"] == 12345

    def test_seed_omitted_when_not_given(self, module, server, make_smi):
        # Warmup calls run_prompt() without a seed; the key must be absent, not null.
        make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            ps.run_prompt()
        assert "seed" not in server.requests[0]["json"]

    def test_samples_power_while_request_runs(self, module, server, make_smi):
        # Counting draws: each sample is 10 W higher than the previous one, so an
        # exact match with the logged mean proves the samples were averaged,
        # not just "the last reading".
        gpus = helpers.default_gpus()
        for gpu in gpus:
            gpu["draw_step"] = 10
        smi = make_smi(gpus)
        server.delay = 0.25
        with helpers.make_power_stat(module, server=server) as ps:
            tps, averages = ps.run_prompt()
        assert tps == 33.3
        # Averages are per GPU, in list order.
        for i, gpu in enumerate(gpus):
            samples = smi.draws(gpu["uuid"])
            assert len(samples) >= 2
            assert averages[i] == int(sum(samples) / len(samples))

    def test_non_200_raises(self, module, server, make_smi):
        make_smi()
        server.status = 500
        with helpers.make_power_stat(module, server=server) as ps:
            with pytest.raises(RuntimeError, match="500"):
                ps.run_prompt()

    def test_missing_timings_raises(self, module, server, make_smi):
        make_smi()
        server.body = {"choices": [{"message": {"content": "hi"}}]}
        with helpers.make_power_stat(module, server=server) as ps:
            with pytest.raises(RuntimeError, match="llama.cpp"):
                ps.run_prompt()


class TestApiKey:
    def test_key_file_not_sent_means_no_auth_header(self, module, server, make_smi):
        make_smi()
        with helpers.make_power_stat(module, server=server) as ps:
            ps.run_prompt()
        assert "authorization" not in server.requests[0]["headers"]

    def test_key_sent_as_bearer_header_with_newline_stripped(self, module, server, make_smi, tmp_path):
        key_file = tmp_path / "key.txt"
        key_file.write_text("sk-test-123\n", encoding="utf-8")
        make_smi()
        with helpers.make_power_stat(module, server=server, api_key_file=str(key_file)) as ps:
            ps.run_prompt()
        assert server.requests[0]["headers"]["authorization"] == "Bearer sk-test-123"

    def test_non_printable_key_rejected_early(self, module, server, make_smi, tmp_path):
        key_file = tmp_path / "key.txt"
        key_file.write_text("bad\nkey\n", encoding="utf-8")
        make_smi()
        with pytest.raises(ValueError, match="non-printable") as excinfo:
            helpers.make_power_stat(module, server=server, api_key_file=str(key_file))
        # The key itself must never appear in the error message.
        assert "bad\nkey" not in str(excinfo.value)
