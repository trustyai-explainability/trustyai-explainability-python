import pytest
from trustyai_xai_bench import run_benchmark_config


@pytest.mark.benchmark(group="xai_bench", min_rounds=1, warmup=False)
def test_level_0(benchmark):
    # ~4.5 min
    result = benchmark(run_benchmark_config, 0)
    benchmark.extra_info['runs'] = result.to_dict('records')


@pytest.mark.skip(reason="full diagnostic benchmark, ~2 hour runtime")
@pytest.mark.benchmark(group="xai_bench", min_rounds=1, warmup=False)
def test_level_1(benchmark):
    result = benchmark(run_benchmark_config, 1)
    benchmark.extra_info['runs'] = result.to_dict('records')


@pytest.mark.skip(reason="very thorough benchmark, >>2 hour runtime")
@pytest.mark.benchmark(group="xai_bench", min_rounds=1, warmup=False)
def test_level_2(benchmark):
    result = benchmark(run_benchmark_config, 2)
    benchmark.extra_info['runs'] = result.to_dict('records')