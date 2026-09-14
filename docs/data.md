# Data sourcing and engineering

Provider documentation checked September 14, 2026. Coverage statements below are vendor claims, not an audit of downloaded observations. No paid subscription has been purchased and no market dataset is bundled.

## Recommended acquisition sequence

1. Use the offline synthetic example to validate mathematics immediately.
2. Build a CSV adapter and a daily SPY pipeline. For exploratory personal research, use Yahoo Finance through the optional yfinance adapter; for a supported commercial source, use an appropriately licensed Alpha Vantage endpoint or a licensed vendor file. Select the provider explicitly; never switch vendors silently after a failure.
3. Acquire hourly SPY for 2005-01-03 through 2020-12-31 to approximate the paper. FirstRate Data is a practical file-based candidate because its SPY product lists coverage from 2000-01-03. Audit a sample's timestamps and adjustment policy before selecting the final source. Alpha Vantage is an API alternative.
4. Extend to separate daily models for QQQ, IWM, TLT, GLD, EEM, and HYG only after checking each series' actual inception and coverage. This is a proposed broad-market research basket, not a historical investable universe. Never backfill an ETF before inception. For cross-sectional stock studies, prefer institutional CRSP access with historical identifiers and inactive securities.

Start with the last complete month, 2026-08-31, as the extension's frozen data cutoff. Preserve the exact first/last available sessions and revision timestamp. A daily proxy cannot reproduce hourly intraday structure, even when it covers the same calendar years.

## Provider comparison

| Source | Best use | Access and limitations | Implementation decision |
| --- | --- | --- | --- |
| [Yahoo via yfinance](https://github.com/ranaroussi/yfinance) | Convenient exploratory daily ETF history | Unofficial interface; project describes research/educational use and Yahoo personal-use restrictions; availability can change | Optional prototype adapter, cached snapshots, explicit adjustment settings; not a promised long-history hourly feed |
| [Alpha Vantage API](https://www.alphavantage.co/documentation/) | Daily adjusted series and historical intraday requests | Daily adjusted and historical intraday endpoints are documented as premium; daily adjusted includes close adjustments and corporate-action events | API key via environment; retrieve full daily series or intraday history month by month; audit symbol coverage |
| [FirstRate Data SPY](https://firstratedata.com/i/etf/SPY) | Paper-period SPY hourly files | Product lists intraday history from 2000-01-03 and multiple bar frequencies, including hourly and daily; paid data | Prefer a licensed static snapshot for reproduction, retain original ZIP checksum and vendor conventions |
| [CRSP US Stock Databases](https://indexes.morningstar.com/research-data-products/crsp-us-stock-databases) | Institutional daily/monthly stock-universe research | Licensed access; active and inactive securities, corporate actions, permanent IDs | Use through institutional delivery/WRDS if entitled; preserve delisting and return semantics |
| [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Long-horizon independent market/factor-return controls | Downloadable research series; portfolios and factors are not SPY prices; historical releases can change | Parse daily files, percent units, missing-value codes, and release metadata |
| [FRED SP500](https://fred.stlouisfed.org/series/SP500) | Context and a limited daily price-index comparison | FRED lists ten years of daily history; SP500 is a price index excluding dividends with specific redistribution restrictions | Do not use as a full-history SPY total-return substitute |
| [Databento historical API](https://databento.com/docs/reference-historical/basics/) | Later intraday, futures, or microstructure extensions | Licensed datasets with dataset-specific coverage and schemas | Query metadata for the desired dataset/date range before assuming suitability; not selected for the 2005 SPY reproduction |

Exact subscription cost depends on coverage, entitlement, and usage. Obtain a current quote before purchase; the architecture works from local licensed files and does not depend on a particular paid subscription. Publicly downloadable does not mean freely redistributable.

## Concrete recipes

### Exploratory daily SPY

Use a provider request equivalent to `symbol=SPY`, `interval=1d`, a start date before the intended training sample, and an exclusive end date after the chosen final session. With yfinance, explicitly request `auto_adjust=False` and corporate actions, keep raw close and adjusted close separately, and calculate log differences on the chosen adjusted series. Inspect the installed adapter's output schema instead of assuming column names or timezone defaults.

Fetch a source snapshot once per experiment. Never repair an old snapshot in place. Validate splits and distribution dates against corporate actions, compare a small random sample and stress periods to a second provider, and record unexplained differences rather than blending series. Prototype adjusted histories downloaded today are revised histories, not guaranteed point-in-time datasets.

### Licensed hourly SPY reproduction

For Alpha Vantage, use `TIME_SERIES_INTRADAY`, `symbol=SPY`, `interval=60min`, `month=YYYY-MM`, `outputsize=full`, explicit adjustment settings, and `extended_hours=false`. Enumerate all months from January 2005 through December 2020; validate returned date ranges instead of trusting HTTP 200. The documentation permits month requests back to January 2000 but specific-symbol coverage still needs checking. Keep credentials out of recorded request URLs. [Endpoint documentation](https://www.alphavantage.co/documentation/).

For FirstRate Data, obtain the licensed SPY file and import through a vendor-specific CSV mapping. Preserve ZIP and member hashes, source timezone, bar timestamp convention, and adjustment version. Its advertised coverage spans the paper period; actual completeness remains a data-quality gate. [SPY product](https://firstratedata.com/i/etf/SPY).

The exchange regular session lasts 6.5 hours on a normal day. Specify whether hourly bins are anchored at 09:30 or on the clock, whether the last partial bar is included, and whether its close-to-close interval is treated identically to a full hour. Treat early closes explicitly. The paper's seven-observation synthetic day is insufficient to infer these details.

A consecutive hourly-close series includes overnight returns at session transitions. The reproduction convention must state whether those returns are included. Run separate overnight/intraday and partial-bar sensitivity analyses; mixing heterogeneous intervals can create apparent regimes. If constructing bars from finer data, keep the aggregation logic and exchange calendar version in the manifest.

### French daily market control

Download the daily three-factor file from the library and retain its dated release. Read `Mkt-RF` and `RF` as daily percent simple returns; reconstruct the broad-market simple return as `(Mkt-RF + RF)/100`, then apply `log1p`. Convert missing-value codes before arithmetic. This is a CRSP-based market control, not the S&P 500 ETF. Record the library's transition from legacy FIZ to CIZ-based research returns beginning with the January 2025 release and use a consistent archived release. [Library and methodology notices](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

### Institutional stock universe

Use permanent security IDs and date-effective membership; do not select today's surviving tickers and project them backward. Import native total returns when available. Handle delisting returns according to the exact CRSP file format/version; do not blindly add a delisting adjustment already incorporated in a provided field. Record symbol changes, corporate actions, and terminal-event exclusions. [CRSP product and user-guide links](https://indexes.morningstar.com/research-data-products/crsp-us-stock-databases).

## Canonical schemas

Store raw downloads unchanged, normalized tables in Parquet, and provenance in JSON. Partition by provider, asset ID, frequency, snapshot ID, and year only when data volume warrants it.

| Table | Required fields |
| --- | --- |
| Bars | `asset_id`, `symbol`, `session_date`, `bar_start_utc`, `bar_end_utc`, `source_timezone`, `open`, `high`, `low`, `close`, nullable `adjusted_close`, nullable `volume`, `currency`, `frequency`, `session_type`, `provider`, `snapshot_id`, `available_at` |
| Corporate actions | `asset_id`, `effective_at`, action type, split ratio or cash distribution, currency, publication/revision metadata, source snapshot |
| Returns | `asset_id`, `interval_start`, `interval_end`, `available_at`, `log_return`, `return_basis`, `is_overnight`, `quality_flags`, source price-row IDs |
| Provenance | Request parameters without secrets, retrieval time, source URL without credentials, original filename, SHA-256, provider version, adjustment method, license reference, calendar version, canonical schema version |

For daily sources that provide only dates, preserve `session_date` and map the actual exchange close to UTC using the exchange calendar. Do not pretend a midnight timestamp is the bar's availability time. If historical availability is unavailable, record an explicit conservative assumed latency and flag the dataset as not verified point-in-time.

Distinguish retrieval time from economic availability time. Store original and revised records as separate snapshots. Any cross-asset joint window must align the same economic intervals, not just loosely matching date labels.

## Quality rules

Reject duplicate keys with conflicting prices; allow identical duplicate rows only through a logged deduplication rule. Sort by time, validate OHLC consistency and nonnegative volume, reject nonfinite or nonpositive prices, and classify expected exchange closures separately from missing bars. Missing bars must not become zero returns through forward-filling.

Keep gaps visible. A missing internal observation invalidates fixed-frequency windows crossing it unless a named irregular-interval policy is selected. Do not drop the missing row and silently treat the return spanning two sessions as a one-session observation. A holiday is not a missing trading session. Record coverage and excluded-window counts by year and asset.

Validate adjusted-series behavior at splits and dividends. Genuine crashes belong in the distribution; suspicious spikes should be quarantined for source comparison rather than winsorized by default. Price levels near futures roll dates or nonpositive futures prices require a different return policy; avoid pretending equity log-return rules work unchanged for futures.

## Operational adapter behavior

Each adapter exposes coverage, bar convention, adjustment support, and rate-limit capabilities. Fetch with bounded exponential backoff, jitter, timeouts, pagination completeness checks, and retry budgets. Recognize provider error payloads returned with HTTP 200. Fail incomplete acquisitions atomically; write to a temporary snapshot and promote only after validation. Log redacted metadata and checksums, not secret-bearing requests.

Use `.env` locally if desired but never commit it. Tests use artificial fixtures or specifically redistributable samples. Provider integration tests are opt-in and credential-gated. Code and documentation are GPL-3.0-only; raw or derived third-party data must satisfy the provider's separate terms before any repository or report publication.
