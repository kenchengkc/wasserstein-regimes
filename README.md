# Wasserstein Regimes

**Market-regime research using optimal transport on full empirical return distributions.**

Instead of representing each rolling market window with a few summary statistics, this project treats the **entire empirical return distribution** as the observation. Windows are compared with Wasserstein distance and clustered around distributional centroids.

The project is motivated by Horvath, Issa, and Muguruza, [Clustering Market Regimes using the Wasserstein Distance](https://arxiv.org/abs/2110.11848v1) (2021). This repository is an independent implementation and research extension, not the authors' code.

## Why this is interesting

Traditional regime models often reduce a return window to volatility, mean, or a small feature vector. That can discard information about skew, tails, and multimodality. Wasserstein clustering instead compares sorted return samples directly, so the clustering objective reflects differences in the full empirical marginal distribution.

For equal-length sorted return vectors `x` and `y` with `w` observations:

```text
W1(x, y)   = mean(abs(x - y))
W2(x, y)^2 = mean((x - y)^2)
```

The corresponding cluster centroids are coordinate-wise medians for W1 and means for squared W2.

## What is implemented

- Exact empirical **W1** and **W2** distances for equal-size samples
- Distributional centroids for both objectives
- A compact Lloyd-style clustering implementation
- Rolling-window construction for return series
- Synthetic variance-switching example with no external data dependency
- Unit tests for the mathematical reference implementation
- Research design covering chronological validation, baselines, data handling, and future multivariate extensions
- MkDocs documentation deployed as a separate static documentation surface

## Current status

**Research prototype / mathematical reference implementation.**

The core distributional clustering machinery is implemented and tested. Market-data adapters, walk-forward empirical evaluation, reporting, and multivariate optimal-transport methods remain planned. The repository does **not** claim predictive alpha, a completed paper replication, or trading performance.

That distinction is intentional: the current code establishes the mathematical and software foundation before adding market-data-dependent empirical claims.

## Research design

The first proposed market experiment uses daily SPY return-distribution windows of 63 observations with stride 5 and compares W1 versus squared W2 across `k=2..6` clusters.

Model selection is designed to happen inside chronological train/validation folds before a final locked evaluation. Planned baselines include:

- volatility-only clustering
- moment-feature clustering
- filtered Gaussian hidden Markov models

The original paper uses hourly SPY data with 35-return windows and 28-return overlap; the daily setup above is an extension rather than a claim of exact replication.

## Important interpretation boundary

Sorting each window preserves its empirical marginal distribution but removes the temporal ordering of returns inside the window. A discovered cluster therefore describes a type of observed return distribution. It does not automatically imply persistence, predict the next regime, or define a trading strategy.

## Repository structure

```text
wasserstein-regimes/
├── src/                  # clustering and distance implementation
├── tests/                # mathematical/unit tests
├── examples/             # offline synthetic example
├── docs/
│   ├── design.md         # mathematical spec and architecture
│   ├── paper-review.md   # paper assumptions and reproduction decisions
│   ├── data.md           # providers, schemas, quality/licensing rules
│   └── implementation-plan.md
├── mkdocs.yml
├── pyproject.toml
└── README.md
```

## Run the reference implementation

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python examples/synthetic.py
```

The synthetic example generates a variance-switching process, fits the clustering retrospectively, and prints cluster membership counts and centroid standard deviations. It is a mathematical smoke test, not an out-of-sample benchmark.

## Documentation

| Document | Purpose |
| --- | --- |
| [Design](docs/design.md) | Mathematical specification, interfaces, architecture, temporal controls, multivariate roadmap |
| [Paper review](docs/paper-review.md) | Experiment settings, evidence, ambiguities, and explicit reproduction decisions |
| [Data sourcing](docs/data.md) | Verified providers, acquisition recipes, schemas, quality checks, and licensing constraints |
| [Implementation plan](docs/implementation-plan.md) | Sequenced milestones, experiments, acceptance criteria, and release gates |

The documentation site is built with MkDocs and can be previewed locally with:

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

## License

Original project code and documentation are licensed under **GNU GPL version 3 only** (`GPL-3.0-only`). Third-party papers and datasets retain their own terms and are not included in this repository.
