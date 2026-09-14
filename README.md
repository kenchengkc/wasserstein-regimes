# Wasserstein Regimes

Market regime research by clustering **entire empirical return distributions** using optimal transport.

Each observation is a rolling window of returns. Sorting its returns preserves the full empirical marginal distribution, including asymmetry, tails, and multimodality. Wasserstein clustering compares these distributions and represents each cluster by a distributional centroid.

Motivated by Horvath, Issa, and Muguruza, [Clustering Market Regimes using the Wasserstein Distance](https://arxiv.org/abs/2110.11848v1) (2021). This is an independent project, not the authors' implementation.

## Project status

**Design and implementation plan, with a tested mathematical reference implementation.** The repository currently contains exact equal-size empirical W1/W2 distances, corresponding median/mean centroids, a small Lloyd clustering implementation, rolling windows, and an offline synthetic example. Market-data adapters, walk-forward evaluation, reports, and multivariate methods are planned, not implemented. No empirical performance or paper replication is claimed.

## Read the plan

| Document | Contents |
| --- | --- |
| [Design](docs/design.md) | Mathematical specification, architecture, interfaces, temporal controls, multivariate roadmap |
| [Paper review](docs/paper-review.md) | Experiment settings, evidence, ambiguities, explicit reproduction decisions |
| [Data sourcing](docs/data.md) | Verified providers, acquisition recipes, schemas, quality and licensing requirements |
| [Implementation plan](docs/implementation-plan.md) | Sequenced milestones, acceptance criteria, experiments, resource estimates, release gates |

## Mathematical contract

For sorted equal-length return vectors `x` and `y` with `w` entries:

```text
W1(x, y)   = mean(abs(x - y))
W2(x, y)^2 = mean((x - y)^2)
```

The W1 objective uses coordinate-wise **medians**. The squared W2 objective uses coordinate-wise **means**. W2 clustering is equivalent to Euclidean k-means on the full sorted vectors, up to a constant factor in the objective. This is distributional clustering even though the implementation uses arrays: the array contains every order statistic, not just selected moments.

These empirical marginal distributions do not retain the temporal order of returns within a window. A cluster describes observed behavior; it does not automatically predict a future regime or imply a trading strategy.

## Run the reference example

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python examples/synthetic.py
```

The example uses no external data or credentials. It prints cluster membership counts and centroids' standard deviations for a synthetic variance-switching process. It fits retrospectively and is a mathematical smoke test, not an out-of-sample benchmark.

## Recommended first experiment

Use daily SPY distribution windows of 63 returns, stride 5, and compare W1 with squared W2 for `k=2..6`. Select settings within chronological training/validation folds, then lock them before final evaluation. Include volatility-only, moment-feature, and filtered Gaussian HMM baselines. The daily experiment is a proposed extension; the paper uses hourly SPY data with 35-return windows and 28-return overlap.

Use the [data guide](docs/data.md) for provider choices. Keep source data and credentials outside version control. Software licensing does not grant rights to redistribute third-party market data.

## Documentation deployment

Vercel serves the documentation as a static site. The root `vercel.json` selects
the **Other** framework preset (`framework: null`), installs the separate docs
dependencies, runs a strict MkDocs build, and publishes `site/`. Keep the Vercel
project root at the repository root so these settings are read.

The Python package is a research library, so it does not define an ASGI/WSGI
entrypoint. Selecting the Python framework preset without this configuration
causes the "No python entrypoint found" error. A future inference API should be
configured as a separate application with an actual HTTP entrypoint.

Build and preview the documentation locally:

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

The documentation build does not run clustering or download market data.

## License

Original project code and documentation are licensed under **GNU GPL version 3 only**, SPDX identifier `GPL-3.0-only`. See [LICENSE](LICENSE). Third-party papers and datasets retain their own terms and are not included.
