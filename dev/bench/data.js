window.BENCHMARK_DATA = {
  "lastUpdate": 1663688232824,
  "repoUrl": "https://github.com/trustyai-explainability/trustyai-explainability-python",
  "entries": {
    "TrustyAI continuous benchmarks": [
      {
        "commit": {
          "author": {
            "email": "ruivieira@users.noreply.github.com",
            "name": "Rui Vieira",
            "username": "ruivieira"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "65d6a43d53924fd311eddfb8e89393d538c20c62",
          "message": "Merge pull request #74 from ruivieira/FAI-828\n\nFAI-828: Continuous benchmark workflow",
          "timestamp": "2022-09-20T16:34:32+01:00",
          "tree_id": "dee5bbc33dc7e15288975a919814769a978ecd91",
          "url": "https://github.com/trustyai-explainability/trustyai-explainability-python/commit/65d6a43d53924fd311eddfb8e89393d538c20c62"
        },
        "date": 1663688232334,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/benchmark.py::test_counterfactual_match",
            "value": 0.8193783428174288,
            "unit": "iter/sec",
            "range": "stddev: 0.012217069303591795",
            "extra": "mean: 1.2204374313354491 sec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/benchmark.py::test_non_empty_input",
            "value": 6.717789062617621,
            "unit": "iter/sec",
            "range": "stddev: 0.009244194693205515",
            "extra": "mean: 148.85849952697754 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/benchmark.py::test_counterfactual_match_python_model",
            "value": 1.5870887391123065,
            "unit": "iter/sec",
            "range": "stddev: 0.0019559752878516743",
            "extra": "mean: 630.084490776062 msec\nrounds: 10"
          }
        ]
      }
    ]
  }
}