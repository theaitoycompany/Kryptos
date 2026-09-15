# Contributing

Keep changes focused and include a synthetic regression for changed privacy behavior.
Do not add production records or a dependency without a concrete need. Report model
misses and false positives separately; never describe template benchmarks as real-world
accuracy. Model changes require pinned revisions and fresh evaluation.

```bash
uv sync --locked
uv run --locked ruff check src tests benchmarks scripts
uv run --locked ruff format --check src tests benchmarks scripts
uv run --locked python -m unittest discover -s tests -q
uv run --locked kryptos selftest
uv run --locked python scripts/check_release.py
```

For browser changes, run `npm ci --ignore-scripts`, `npx playwright install chromium`,
`npm run format:check`, and `npm test`. For packaging changes, build both distributions
and run `python scripts/check_distribution.py` in the development environment.

Contributions are licensed under Apache-2.0. Use the repository issue tracker for
reproducible, non-sensitive reports and explain the behavior and validation in pull requests.
