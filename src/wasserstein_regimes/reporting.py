# SPDX-License-Identifier: GPL-3.0-only
"""Read-only, local report generation from completed study artifacts."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_NAMES = {'w2': 'raw W2', 'w1': 'raw W1', 'shape_w2': 'shape W2',
          'volatility': 'volatility', 'mean_vol': 'mean + volatility',
          'moments': 'moments', 'rich': 'rich features', 'gmm': 'Gaussian mixture',
          'hmm': 'Gaussian HMM'}
_COLORS = ('#275dad', '#bf5b17', '#238763', '#873ca0', '#a42e3d')


def _e(value):
    return html.escape(str(value), quote=True)


def _n(value, digits=3):
    if value is None:
        return '—'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _e(value)
    return f'{number:.{digits}f}' if np.isfinite(number) else '—'


def _cells(values, header=False):
    tag = 'th' if header else 'td'
    return '<tr>' + ''.join(f'<{tag}>{_e(v)}</{tag}>' for v in values) + '</tr>'


def _table(headers, rows):
    return '<div class="scroll"><table><thead>' + _cells(headers, True) + '</thead><tbody>' + ''.join(
        _cells(row) for row in rows) + '</tbody></table></div>'


def _figure(path, draw, title, alt):
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=130)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    draw(ax)
    ax.set_title(title, loc='left', fontsize=12, color='#182638', pad=12)
    ax.grid(axis='y', color='#e5eaf0', linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors='#37475a', labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor='white', metadata={'Software': 'wasserstein-regimes'})
    plt.close(fig)
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{_e(alt)}"><figcaption>{_e(alt)}</figcaption></figure>'


def _select(frame, fold, model):
    return frame.loc[(frame['fold'] == fold) & (frame['model'] == model) &
                     (frame['policy'] == 'strict')].sort_values('date')


def _plot_centroids(ax, centers):
    x = (np.arange(centers.shape[1]) + .5) / centers.shape[1]
    for i, center in enumerate(centers):
        ax.plot(x, center, color=_COLORS[i % len(_COLORS)], lw=2, label=f'state {i}')
    ax.set(xlabel='probability midpoint', ylabel='sorted daily log return')
    ax.legend(frameon=False, ncol=min(len(centers), 5), fontsize=8)


def _plot_timeline(ax, frame, fold):
    ticks, tick_names = [], []
    width = int(frame.loc[frame['fold'] == fold, 'regime'].max()) + 1
    for offset, model in enumerate(('w2', 'shape_w2', 'volatility')):
        rows = _select(frame, fold, model)
        if rows.empty:
            continue
        dates = pd.to_datetime(rows['date'])
        labels = rows['regime'].to_numpy(dtype=float)
        ax.scatter(dates, np.full(len(rows), offset * width) + labels,
                   s=9, marker='s', color=_COLORS[offset], alpha=.8, label=_NAMES[model])
        for state in sorted(rows['regime'].unique()):
            ticks.append(offset * width + int(state))
            tick_names.append(f'{_NAMES[model]} · {state}')
    ax.set_yticks(ticks, tick_names)
    ax.set_ylabel('model · occupied state')
    ax.set_xlabel('strict test assignment date')
    ax.tick_params(axis='x', rotation=20)


def _plot_occupancy(ax, summary):
    values = summary.get('occupancy', [])
    ax.bar(np.arange(len(values)), values, color=_COLORS[:len(values)])
    ax.set(xlabel='raw W2 state', ylabel='fraction of strict scored windows', ylim=(0,1))
    ax.set_xticks(np.arange(len(values)))


def _plot_components(ax, summary):
    between = summary.get('centroid_decomposition', {})
    within = summary.get('assignment_decomposition', {})
    for i, component in enumerate(('location', 'scale', 'shape')):
        ax.barh([0,1], [between.get(component, 0), within.get(component, 0)],
                left=[sum(between.get(c, 0) for c in ('location','scale','shape')[:i]),
                      sum(within.get(c, 0) for c in ('location','scale','shape')[:i])],
                color=_COLORS[i], label=component)
    ax.set_yticks([0,1], ['between centroids','within assignments'])
    ax.invert_yaxis()
    ax.set(xlabel='share of summed squared W2', xlim=(0,1))
    ax.legend(frameon=False, ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(.5,-.2))


def _plot_novelty(ax, rows):
    dates = pd.to_datetime(rows['date'])
    ax.plot(dates, rows['novelty_percentile'].to_numpy(dtype=float),
            color=_COLORS[0], linewidth=1.2)
    ax.axhline(.99, color=_COLORS[4], linestyle='--', linewidth=1.5, label='99th percentile')
    ax.set(xlabel='strict test assignment date', ylabel='validation distance percentile', ylim=(0,1.03))
    ax.tick_params(axis='x', rotation=20)
    ax.legend(frameon=False, fontsize=8)


def _plot_stability(ax, stability):
    models = stability.get('models', {})
    names = [name for name in ('w2','shape_w2','volatility','moments') if name in models]
    x = np.arange(len(names))
    for offset, (key, label) in enumerate((('seed','seed repeats'), ('block_bootstrap','block resampling'))):
        values = [models[name].get(key, {}).get('ari_mean', np.nan) for name in names]
        ax.bar(x + (offset-.5)*.35, values, width=.35, color=_COLORS[offset], label=label)
    ax.set_xticks(x, [_NAMES.get(name,name) for name in names])
    ax.set(xlabel='model', ylabel='mean validation ARI', ylim=(0,1))
    ax.legend(frameon=False, fontsize=8)


def _plot_overlap(ax, overlap):
    rows = overlap.get('rows', [])
    x = np.arange(len(rows))
    ax.bar(x-.18, [row.get('observed_mean_dwell', np.nan) for row in rows], .36,
           label='observed', color=_COLORS[0])
    ax.bar(x+.18, [row.get('null_mean_dwell', np.nan) for row in rows], .36,
           label='IID null mean', color=_COLORS[1])
    ax.set_xticks(x, [str(row.get('fit_stride', '—')) for row in rows])
    ax.set(xlabel='training fit stride (validation scored daily)', ylabel='mean dwell, windows')
    ax.legend(frameon=False, fontsize=8)


def _optional_json(path, title):
    if not path.exists():
        return ''
    value = json.loads(path.read_text())
    rows = []
    def walk(prefix, item):
        if isinstance(item, dict):
            for key in sorted(item):
                walk(f'{prefix} / {key}' if prefix else str(key), item[key])
        elif isinstance(item, list):
            if item and all(isinstance(v, (int,float)) or v is None for v in item):
                rows.append((prefix, ', '.join(_n(v) for v in item)))
            else:
                rows.append((prefix, json.dumps(item, sort_keys=True)))
        else:
            rows.append((prefix, _n(item) if isinstance(item, (int,float)) else str(item)))
    walk('', value)
    return f'<section><h2>{_e(title)}</h2><p>Saved measurements; definitions and sample size follow the saved JSON fields.</p>' + _table(['field','saved value'], rows) + '</section>'


def _synthetic_section(root, config, chart):
    path = root/'synthetic.json'
    if not path.exists():
        return ''
    value = json.loads(path.read_text())
    controls = _synthetic_controls(value) if isinstance(value, dict) else ''
    if not isinstance(value, dict) or not isinstance(value.get('recovery'), dict):
        return _optional_json(path, 'Synthetic recovery')
    records = []
    for suite, lengths in sorted(value['recovery'].items()):
        for length, study in sorted(lengths.items(), key=lambda pair: int(pair[0])):
            for method, measurements in sorted(study.get('methods', {}).items()):
                delay = measurements.get('detection_delay', {})
                records.append(dict(suite=suite, length=length, method=method,
                                    pure=measurements.get('pure', {}).get('ari', {}).get('mean'),
                                    mixed=measurements.get('mixed', {}).get('ari', {}).get('mean'),
                                    delay=delay.get('detected_only_delay', {}).get('mean'),
                                    detected=delay.get('detected'), censored=delay.get('censored'),
                                    repeats=study.get('repetitions'), stride=study.get('score_stride')))
    if not records:
        return _optional_json(path, 'Synthetic recovery')
    featured = str(config.get('window_length', records[0]['length']))
    if not any(row['length'] == featured for row in records):
        featured = records[0]['length']
    selected = [row for row in records if row['length'] == featured]
    suites = sorted({row['suite'] for row in selected})
    methods = sorted({row['method'] for row in selected})
    def bars(ax, key, ylabel):
        x = np.arange(len(suites))
        width = .8/max(len(methods),1)
        for i, method in enumerate(methods):
            values = [next((r[key] for r in selected if r['suite'] == suite and r['method'] == method), np.nan)
                      for suite in suites]
            ax.bar(x-.4+width*(i+.5), [np.nan if v is None else v for v in values],
                   width, color=_COLORS[i % len(_COLORS)], label=_NAMES.get(method, method))
        ax.set_xticks(x, suites, rotation=12)
        ax.set(xlabel='synthetic contrast', ylabel=ylabel)
        ax.legend(frameon=False, fontsize=8, ncol=min(len(methods), 4))
    rows = [(r['suite'], r['length'], _NAMES.get(r['method'], r['method']), r['repeats'], r['stride'],
             _n(r['pure']), _n(r['mixed']), _n(r['delay']), r['detected'], r['censored']) for r in records]
    note = value.get('inference_note', '')
    shape_suites = [suite for suite in sorted(value['recovery']) if suite != 'variance']
    def by_length(ax, key, ylabel):
        ax.set_visible(False)
        for panel, suite in enumerate(shape_suites):
            axis = ax.figure.add_subplot(1, len(shape_suites), panel+1)
            suite_rows = [row for row in records if row['suite'] == suite]
            for i, method in enumerate(sorted({r['method'] for r in suite_rows})):
                points = sorted([r for r in suite_rows if r['method'] == method], key=lambda r: int(r['length']))
                axis.plot([int(r['length']) for r in points],
                          [np.nan if r[key] is None else r[key] for r in points], marker='o', markersize=3,
                          color=_COLORS[i % len(_COLORS)], linestyle=('-', '--')[i//len(_COLORS) % 2],
                          linewidth=1.3, label=_NAMES.get(method,method))
            axis.set_title(suite.replace('_',' '), fontsize=10)
            axis.set_xticks(sorted({int(row['length']) for row in suite_rows}))
            axis.set_xlabel('window length, observations', fontsize=8)
            axis.set_ylabel(ylabel if panel == 0 else '', fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(axis='y', color='#e5eaf0')
            axis.spines[['top','right']].set_visible(False)
            axis.legend(frameon=False, fontsize=6, ncol=2)
    tradeoff = ''
    if shape_suites:
        tradeoff = ('<h3>Recovery and delay versus window length</h3><p>Shape-changing contrasts show saved '
                    'pure-window recovery and first correct post-switch assignment delay across L. Delay is detected-only; '
                    'it is not a persistence-confirmed change detector. Score stride sets raw-observation resolution, '
                    'and censor counts remain in the table below.</p>'
                    + chart('synthetic_recovery_by_length', lambda ax: by_length(ax,'pure','mean pure-window ARI'),
                            'Synthetic recovery versus window length',
                            'Shape-changing synthetic contrasts: pure-window ARI versus window length')
                    + chart('synthetic_delay_by_length', lambda ax: by_length(ax,'delay','detected-only delay, observations'),
                            'Synthetic delay versus window length',
                            'Shape-changing synthetic contrasts: detected-only transition delay versus window length'))
    return ('<section><h2>Synthetic recovery</h2><p>Saved independent-stream synthetic measurements. '
            'Pure-window and mixed-window ARI are shown separately. Transition delay averages include detected events only; '
            'censored transitions are counted separately and must not be read as zero delay. '
            f'Charts use window length {_e(featured)}; the table includes all saved lengths. {_e(note)}</p>'
            + chart('synthetic_recovery', lambda ax: bars(ax,'pure','mean pure-window ARI'),
                    f'Synthetic pure-window recovery, L={featured}',
                    f'Saved mean pure-window synthetic recovery ARI at length {featured}')
            + chart('synthetic_delay', lambda ax: bars(ax,'delay','mean detected-only delay, observations'),
                    f'Synthetic detected-only delay, L={featured}',
                    f'Saved mean detected-only synthetic transition delay at length {featured}')
            + tradeoff
            + _table(['contrast','window length','method','repeats','score stride','pure ARI','mixed ARI',
                      'detected-only delay','detected','censored'], rows) + '</section>' + controls)


def _synthetic_controls(value):
    sections = []
    exact = value.get('exact_moments', {})
    if exact:
        numeric = lambda key: f'{float(exact[key]):.12e}' if exact.get(key) is not None else '—'
        methods = exact.get('methods') or {
            'w2': {'test_ari': exact.get('w2_test_ari')},
            'moments': {'effective_clusters': exact.get('moments_effective_clusters')}}
        rows = [(_NAMES.get(name,name), _n(result.get('test_ari')),
                 _n(result.get('test_balanced_accuracy')), result.get('effective_clusters','—'))
                for name, result in sorted(methods.items())]
        sections.append('<section><h2>Exact finite-moment blocks</h2><p>These are disjoint '
                        f'{_e(exact.get("block_length", "—"))}-atom empirical blocks. The exact-block first-four-moment '
                        f'maximum measured gap is {numeric("max_first_four_moment_gap")}; saved W2 between laws is '
                        f'{numeric("w2_between_laws")}. This finite counterexample is separate from iid rolling windows '
                        'whose distributions match only population moments. Effective cluster counts show baseline '
                        'collapse where measured; test recovery uses the saved training-derived label mapping.</p>'
                        f'<p>Saved scope: {_e(exact.get("scope", "unspecified"))}.</p>'
                        + _table(['method','test ARI','test balanced accuracy','effective clusters'],rows) + '</section>')
    stationary = value.get('stationary', {})
    if stationary:
        rows = []
        for length, study in sorted(stationary.items(), key=lambda pair: int(pair[0])):
            for name, result in sorted(study.get('methods', {}).items()):
                row = [length, _NAMES.get(name,name), study.get('repetitions','—'),
                       study.get('score_stride','—'), result.get('forced_k','—')]
                row.extend(_n(result.get(key, {}).get('mean')) for key in (
                    'seed_ari','switch_frequency','mean_dwell_scored_windows',
                    'mean_dwell_observations_approx','novelty_false_flag_rate','centroid_w2_distance'))
                row.extend(result.get(key, 'not saved') for key in (
                    'calibration_segment','novelty_distance_metric','novelty_distance_units'))
                rows.append(row)
        sections.append('<section><h2>Stationary forced-clustering control</h2><p>A stationary single-law stream '
                        'is forced into K states. Persistence, seed stability, and false-novelty means below are guardrails: '
                        'stable or persistent clusters can occur without switching generating laws. Dwell counts scored windows; '
                        'its raw-observation conversion is approximate. False-novelty calibration and distance geometry are '
                        'reported as saved; missing definitions remain marked as not saved. These independent-stream means '
                        'are descriptive, not significance tests.</p>'
                        + _table(['L','method','repeats','score stride','forced K','seed ARI','switch frequency',
                                  'dwell scored windows','dwell observations (approx.)','false-novelty rate',
                                  'centroid W2','calibration','novelty metric','novelty units'],rows) + '</section>')
    return ''.join(sections)


def report(run_path) -> Path:
    """Render a deterministic, self-contained HTML report without fitting models."""
    root = Path(run_path)
    config = json.loads((root/'config.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    metrics = json.loads((root/'metrics.json').read_text())
    with np.load(root/'centroids.npz') as archive:
        centers = np.asarray(archive['centers'], dtype=float)
    frame = pd.read_parquet(root/'assignments.parquet')
    folds = sorted(metrics['folds'], key=lambda row: row['year'])
    latest = folds[-1]
    year = latest['year']
    raw = latest['models']['w2']['strict']
    raw_rows = _select(frame, year, 'w2')
    if raw_rows.empty or centers.ndim != 2 or len(centers) != latest['k']:
        raise ValueError('Latest strict raw W2 rows and centroids must match the selected fold')
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    chart = lambda name, draw, title, alt: _figure(figures/f'{name}.png', draw, title, alt)
    bounds = latest['boundaries']
    stage = config.get('stage', 'development')
    parts = [f'<header><p class="eyebrow">LOCAL RESEARCH ARTIFACT · {_e(stage)}</p>'
             f'<h1>Distributional regime study</h1><p class="lede">Run {_e(manifest.get("run_id", root.name))} · {year} strict test fold · {len(raw_rows)} scored windows</p></header>']
    parts.append('<section><h2>Dataset and temporal contract</h2>'
                 f'<p>The input is {_e(config.get("dataset", "unspecified"))}, supplied by {_e(config.get("provider", manifest.get("provider", "unspecified")))}. '
                 f'Return basis: {_e(manifest.get("return_basis", "unspecified"))}. '
                 f'Window length { _e(config.get("window_length", "—")) } sessions; training fit stride {_e(config.get("fit_stride", "—"))}; '
                 f'scoring stride {_e(config.get("score_stride", "—"))}. '
                 f'Training ends {_e(bounds["train_end"])}, validation ends {_e(bounds["validation_end"])}, '
                 f'and this test fold ends {_e(bounds["test_end"])}.</p>'
                 '<p>Strict windows purge shared raw price intervals across split boundaries. Daily test assignments still overlap. '
                 'Historical adjusted prices can be revised later; the manifest does not claim point-in-time data.</p></section>')
    all_fold_rows = []
    for fold in folds:
        selection = next((r.get('validation_silhouette') for r in fold.get('selection', []) if r.get('k') == fold['k']), None)
        refit = fold.get('refit', {})
        all_fold_rows.append((fold['year'], fold['boundaries']['train_end'], fold['boundaries']['validation_end'],
                              fold['boundaries']['test_end'], fold['k'], _n(selection),
                              _n(refit.get('ari')) if refit else '—'))
    parts.append('<section><h2>Fold model selection</h2><p>K was selected by validation silhouette in raw full-quantile W2 geometry. '
                 'This measures separation, not predictive value. Each fold trains on its own expanding history.</p>'
                 + _table(['fold','train end','validation end','test end','K','selected validation silhouette','refit ARI'], all_fold_rows)
                 + '<h3>Latest fold K candidates</h3>'
                 + _table(['candidate K','validation silhouette','converged'],
                          [(r.get('k','—'), _n(r.get('validation_silhouette')),
                            r.get('converged','—')) for r in latest.get('selection', [])])
                 + ('<p>The selected K changed across folds; state numbers do not denote the same regimes across those folds.</p>'
                    if len({f['k'] for f in folds}) > 1 else '') + '</section>')
    parts.append('<section><h2>Raw W2 quantile barycenters</h2><p>Each curve is a saved quantile barycenter, '
                 'evaluated at probability midpoints. The ordinate is the sorted daily log-return value.</p>'
                 + chart('centroids', lambda ax: _plot_centroids(ax,centers), 'Raw W2 centroids',
                         'Saved raw W2 quantile centroid curves at probability midpoints') + '</section>')
    parts.append('<section><h2>Strict daily assignment timeline</h2><p>Rows show model-specific state IDs in the latest fold. '
                 'Shape W2 uses standardized windows; volatility uses a feature assignment. State IDs are not aligned between models.</p>'
                 + chart('timeline', lambda ax: _plot_timeline(ax,frame,year), 'Daily strict test assignments',
                         'Latest fold strict daily raw W2, shape W2, and volatility state assignments') + '</section>')
    parts.append('<section><h2>Occupancy and W2 geometry</h2><p>Occupancy counts overlapping strict scored windows. '
                 'Between-centroid location, scale, and shape shares describe pairwise saved raw W2 centroids; '
                 'within-assignment shares describe windows assigned to those raw W2 centroids. '
                 'Both shares partition summed squared W2 and answer different questions.</p>'
                 + chart('occupancy', lambda ax: _plot_occupancy(ax,raw), 'Raw W2 state occupancy',
                         'Fraction of strict latest-fold windows in each raw W2 state')
                 + chart('components', lambda ax: _plot_components(ax,raw), 'W2 squared-distance components',
                         'Location, scale, and shape shares between centroids and within assignments')
                 + f'<p>Mean within-assignment raw W2: {_n(raw.get("within_w2"))}; mean between-centroid W2: '
                   f'{_n(raw.get("between_centroid_w2_mean"))}. These are different populations of distances. '
                   f'Strict mean dwell: {_n(raw.get("temporal", {}).get("mean_dwell"))} scored windows; '
                   f'switching frequency: {_n(raw.get("temporal", {}).get("switching_frequency"))} among adjacent scored windows.</p></section>')
    baseline_rows = []
    for name, model in latest['models'].items():
        result = model.get('strict', {})
        comps = result.get('assignment_decomposition', {})
        units = 'standardized window units' if model.get('shape_standardized') else 'return units'
        baseline_rows.append((_NAMES.get(name,name), result.get('n','—'), _n(latest.get('ari_against_raw_w2',{}).get(name)),
                              _n(result.get('within_w2')), units,
                              '/'.join(_n(comps.get(c),2) for c in ('location','scale','shape'))))
    parts.append('<section><h2>Baseline comparison</h2><p>ARI compares strict assignments with raw W2 on the same fold. '
                 'Location/scale/shape are shares of W2 distance to each model’s saved prototypes, where available. '
                 'Within W2 is not comparable between shape-standardized and return units. Feature and HMM assignments '
                 'are not necessarily nearest-prototype assignments; their novelty is W2 distance to prototypes, '
                 'not assignment probability. HMM prototypes are Gaussian emission quantiles.</p>'
                 + _table(['model','strict n','ARI vs raw W2','mean within W2','distance units','location/scale/shape shares'], baseline_rows)
                 + '</section>')
    outcome_rows = []
    for state in raw.get('conditional_outcomes', []):
        row = [state.get('regime','—'), state.get('n','—'), _n(state.get('occupancy'))]
        for key in ('forward_log_return','forward_volatility','forward_drawdown','forward_downside'):
            item = state.get(key, {})
            interval = item.get('block_interval')
            row.append(f'{_n(item.get("mean"))} [{_n(interval[0])}, {_n(interval[1])}]' if interval else _n(item.get('mean')))
        outcome_rows.append(row)
    parts.append('<section><h2>Future 21-session outcomes by state</h2><p>These are descriptive means after each assignment endpoint; '
                 'the outcomes never enter fitting. Brackets are moving-block intervals over overlapping assignments, '
                 'not significance tests. End-of-series windows without a complete future horizon are omitted per outcome.</p>'
                 + _table(['raw W2 state','assigned n','occupancy','forward log return','annualized volatility',
                           'drawdown','annualized downside'], outcome_rows) + '</section>')
    parts.append('<section><h2>Novelty percentile</h2><p>The line is the percentile of nearest-prototype '
                 'distance against validation calibration distances. The 99th percentile is the saved novelty threshold; '
                 'the percentile is not a probability of a regime or future event. '
                 f'Strict raw W2 threshold exceedance rate: {_n(raw.get("ood_rate"))}.</p>'
                 + chart('novelty', lambda ax: _plot_novelty(ax,raw_rows), 'Raw W2 novelty percentile',
                         'Nearest-prototype validation distance percentile with 99th-percentile threshold') + '</section>')
    stability = metrics.get('stability', {})
    if stability.get('models'):
        stab_rows = []
        for name, groups in stability['models'].items():
            for key in ('seed','block_bootstrap'):
                group = groups.get(key, {})
                stab_rows.append((_NAMES.get(name,name), key.replace('_',' '), group.get('repetitions','—'),
                                  _n(group.get('ari_mean')), ', '.join(_n(v) for v in group.get('ari_range', []))))
        parts.append('<section><h2>Validation-only stability</h2><p>Seed and training-return block repetitions compare validation assignments '
                     'with a fixed validation reference. ARI ranges are observed repeat ranges, not confidence intervals.</p>'
                     + chart('stability', lambda ax: _plot_stability(ax,stability), 'Validation assignment stability',
                             'Seed and block-resampled mean validation ARI for raw W2, shape W2, volatility, and moments')
                     + _table(['model','perturbation','repeats','mean ARI','observed range'], stab_rows) + '</section>')
    overlap = metrics.get('overlap', {})
    if overlap.get('rows'):
        overlap_rows = [(r.get('fit_stride','—'), _n(r.get('observed_mean_dwell')),
                         _n(r.get('null_mean_dwell')), ', '.join(_n(v) for v in r.get('null_interval',[])),
                         _n(r.get('ari_vs_primary'))) for r in overlap['rows']]
        parts.append('<section><h2>Overlapping-window null</h2><p>Observed validation mean dwell is compared with the '
                     'saved IID-resampled development-marginal null while overlap is retained. This is a descriptive '
                     'fit-stride sensitivity comparison, not evidence of an independent temporal signal.</p>'
                     + chart('overlap', lambda ax: _plot_overlap(ax,overlap), 'Validation dwell and overlap null',
                             'Observed and null mean validation dwell across training fit strides')
                     + _table(['fit stride','observed dwell','null mean dwell','null interval','ARI vs primary'], overlap_rows) + '</section>')
    parts.append(_synthetic_section(root,config,chart))
    parts.append(_optional_json(root/'benchmark.json','Benchmark'))
    parts.append('<section><h2>Limitations</h2><p>One ETF and one historical holdout cannot establish general economic value. '
                 'Overlapping scored windows are dependent; state labels can change after refits. The report uses saved '
                 'measurements only and provides no trading claim or significance claim. Real source data remain in local artifacts.</p></section>')
    parts.append('<section><h2>Full manifest</h2><p>Provenance and temporal metadata as saved with this run.</p>'
                 f'<pre>{_e(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False))}</pre></section>')
    style = '''body{margin:0;background:#fff;color:#182638;font:16px/1.56 system-ui,-apple-system,Segoe UI,sans-serif}
    main{max-width:1050px;margin:auto;padding:40px 32px 80px}header{border-bottom:3px solid #244b79;padding-bottom:24px}
    h1{font-size:2.3rem;line-height:1.2;margin:.3rem 0}h2{font-size:1.35rem;margin:0 0 .7rem;color:#244b79}
    section{padding:30px 0;border-bottom:1px solid #dce4ec}.eyebrow{color:#50677f;font-size:.78rem;letter-spacing:.12em;font-weight:700}
    .lede{color:#43596f;font-size:1.08rem}p{max-width:900px;margin:.55rem 0 1rem}figure{margin:18px 0 28px}
    img{display:block;width:100%;height:auto;max-width:1000px}figcaption{color:#50677f;font-size:.84rem;margin-top:7px}
    .scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border-bottom:1px solid #dce4ec;padding:9px;text-align:left;vertical-align:top}
    th{background:#edf2f7;color:#203e60}tbody tr:nth-child(even){background:#f8fafc}pre{overflow-x:auto;background:#f3f6f9;padding:20px;font-size:.8rem}
    @media(max-width:650px){main{padding:24px 16px}h1{font-size:1.8rem}}'''
    output = root/'report.html'
    output.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                      '<title>Distributional regime study</title><style>' + style + '</style></head><body><main>'
                      + ''.join(parts) + '</main></body></html>\n', encoding='utf-8')
    return output
