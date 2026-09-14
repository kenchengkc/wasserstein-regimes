# Wasserstein Regimes

Discover recurring market conditions by clustering **entire empirical return distributions** with optimal transport.

Each observation is a rolling window of returns. Its sorted values retain the full empirical marginal distribution, including asymmetry, tails, and multimodality. Wasserstein clustering compares these distributions and represents each cluster with a distributional centroid.

## Explore the project

| Guide | What you will find |
| --- | --- |
| [System design](design.md) | Mathematical specification, package architecture, temporal controls, and multivariate extensions |
| [Data sourcing](data.md) | Provider comparisons, acquisition recipes, canonical schemas, and data-quality rules |
| [Implementation plan](implementation-plan.md) | Sequenced milestones, experiments, acceptance criteria, and resource estimates |
| [Paper review](paper-review.md) | The motivating experiments, mathematical ambiguities, and reproduction decisions |

## The mathematical foundation

For sorted equal-length samples `x` and `y`:

```text
W1(x, y)   = mean(abs(x - y))
W2(x, y)^2 = mean((x - y)^2)
```

The W1 objective uses coordinate-wise median centroids. The squared W2 objective uses coordinate-wise mean centroids. Each sample contains every observed return in the window, rather than a handful of summary statistics.

Sorting discards the temporal order within each window. These models describe empirical marginal distributions; joint dependence and return ordering require the extensions described in the design.

## Current status

The repository provides a design and implementation plan, a tested mathematical reference implementation, and an offline synthetic example. Market-data ingestion, walk-forward evaluation, and the full research reporting pipeline remain planned. This site publishes the documentation; it does not run market inference or accept uploaded data.

## Run the reference example locally

After cloning the repository, use Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python examples/synthetic.py
```

The example uses artificial returns and requires no credentials. It fits retrospectively and makes no out-of-sample performance claim.

## Research and license

Motivated by Horvath, Issa, and Muguruza, [Clustering Market Regimes using the Wasserstein Distance](https://arxiv.org/abs/2110.11848v1) (2021). This is an independent implementation.

Original project code and documentation are licensed under [GNU GPL version 3 only](https://github.com/kenchengkc/wasserstein-regimes/blob/main/LICENSE). Third-party papers and market datasets retain their own terms. Repository links require access while the repository is private.
