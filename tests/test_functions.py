"""Tests for the pure functions: to_base_url, csv_header, parse_tps, gen_power_steps."""

import pytest


class TestToBaseUrl:
    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            ("localhost:8080", "http://localhost:8080"),
            ("127.0.0.1:8080", "http://127.0.0.1:8080"),
            ("http://host:8080", "http://host:8080"),
            ("http://host:8080/", "http://host:8080"),
            ("http://host:8080/v1", "http://host:8080"),
            ("http://host:8080/v1/", "http://host:8080"),
            ("https://host:8443/v1", "https://host:8443"),
        ],
    )
    def test_to_base_url(self, module, endpoint, expected):
        assert module.to_base_url(endpoint) == expected


class TestCsvHeader:
    def test_csv_header_matches_row_layout(self, module):
        assert module.csv_header(1) == [
            "Timestamp",
            "GPU0 power limit W",
            "GPU0 avg power W",
            "TPS",
        ]
        assert module.csv_header(2) == [
            "Timestamp",
            "GPU0 power limit W",
            "GPU1 power limit W",
            "GPU0 avg power W",
            "GPU1 avg power W",
            "TPS",
        ]
        assert module.csv_header(3) == [
            "Timestamp",
            "GPU0 power limit W",
            "GPU1 power limit W",
            "GPU2 power limit W",
            "GPU0 avg power W",
            "GPU1 avg power W",
            "GPU2 avg power W",
            "TPS",
        ]


class TestParseTps:
    def test_valid(self, module):
        assert (
            module.LLMPowerStat.parse_tps({"timings": {"predicted_per_second": 33.3}})
            == 33.3
        )
        assert module.LLMPowerStat.parse_tps({"timings": {"predicted_per_second": 42}}) == 42.0

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"timings": {}},
            {"timings": {"predicted_per_second": None}},
        ],
    )
    def test_missing_or_invalid_raises(self, module, body):
        with pytest.raises(RuntimeError, match="llama.cpp"):
            module.LLMPowerStat.parse_tps(body)


class TestGenPowerSteps:
    @staticmethod
    def gpu(module, default: int, minimum: int):
        return module.GPU(
            model_name="fake",
            uuid=f"GPU-{default}-{minimum}",
            power_default_limit=default,
            power_min_limit=minimum,
        )

    def steps(self, module, gpus, step: int) -> list[list[int]]:
        # gen_power_steps only reads self.power_step_W (and recurses into
        # itself), so an instance created without __init__ is enough.
        owner = module.LLMPowerStat.__new__(module.LLMPowerStat)
        owner.power_step_W = step
        # run_analysis() passes a copy; gen_power_steps consumes the list it is given.
        return list(owner.gen_power_steps(list(gpus)))

    def test_two_gpus_agents_md_example(self, module):
        gpus = [self.gpu(module, 300, 200), self.gpu(module, 250, 150)]
        assert self.steps(module, gpus, 100) == [
            [300, 250],
            [200, 250],
            [300, 150],
            [200, 150],
        ]

    def test_single_gpu(self, module):
        gpus = [self.gpu(module, 300, 200)]
        assert self.steps(module, gpus, 100) == [[300], [200]]

    def test_step_larger_than_range(self, module):
        gpus = [self.gpu(module, 300, 200)]
        assert self.steps(module, gpus, 1000) == [[300]]

    def test_default_equal_to_min(self, module):
        gpus = [self.gpu(module, 300, 300)]
        assert self.steps(module, gpus, 50) == [[300]]

    def test_three_gpus_order_and_bounds(self, module):
        gpus = [
            self.gpu(module, 300, 200),  # 300, 200
            self.gpu(module, 250, 150),  # 250, 150
            self.gpu(module, 300, 200),  # 300, 200
        ]
        steps = self.steps(module, gpus, 100)
        # The default limit is the first value on every axis.
        assert steps[0] == [300, 250, 300]
        # GPU 0 changes fastest: its value alternates every step.
        assert [s[0] for s in steps] == [300, 200] * 4
        # 2 * 2 * 2 combinations, all within [min, default].
        assert len(steps) == 8
        allowed = [
            {300, 200},
            {250, 150},
            {300, 200},
        ]
        for step in steps:
            for i, value in enumerate(step):
                assert value in allowed[i]

    def test_input_list_survives_when_a_copy_is_passed(self, module):
        gpus = [self.gpu(module, 300, 200)]
        self.steps(module, gpus, 100)
        assert len(gpus) == 1
