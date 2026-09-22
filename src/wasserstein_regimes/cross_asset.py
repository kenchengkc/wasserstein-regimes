# SPDX-License-Identifier: GPL-3.0-only
"""Independent asset replication and comparison of saved research artifacts."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

from .clustering import WassersteinKMeans
from .data import load_csv
from .evaluation import align_centroids
from .experiments import artifact_id, run, split_masks, verify_artifact, write_json
from .reporting import report
from .transport import pairwise_distance, standardize_windows
from .windows import make_windows


def validate_protocol(configs):
    """Only provenance and asset identifiers may differ within the replication."""
    if not configs:
        raise ValueError('No asset configurations supplied')
    ignored = {'symbol', 'dataset', 'dataset_sha256'}
    baseline = next(iter(configs.values()))
    for symbol, config in configs.items():
        for key in (set(baseline) | set(config)) - ignored:
            if baseline.get(key) != config.get(key):
                raise ValueError(f'Asset-specific protocol change for {symbol}: {key}')


def compare_refits(previous, current, samples, *, shape_only):
    """Compare frozen models on identical observations; never fit either model."""
    x = standardize_windows(samples) if shape_only else np.asarray(samples)
    a, b = previous.predict(x), current.predict(x)
    ka, kb = len(previous.centers_), len(current.centers_)
    mapping, displacement = None, None
    if ka == kb:
        mapping = align_centroids(previous.centers_, current.centers_)
        displacement = float(pairwise_distance(current.centers_, previous.centers_)[np.arange(kb), mapping].mean())
    na, nb = len(np.unique(a)), len(np.unique(b))
    return dict(ari=float(adjusted_rand_score(a,b)), previous_k=ka, current_k=kb,
                previous_occupied=na,current_occupied=nb,both_single_state=na==nb==1,
                mapping=None if mapping is None else mapping.tolist(),
                centroid_displacement=displacement,n=len(x))


def refit_history(development, holdout):
    """Load saved raw/shape models and compare on each later validation period."""
    paths = [Path(development), Path(holdout)]
    records = []
    for path in paths:
        verify_artifact(path)
        config = json.loads((path/'config.json').read_text())
        metrics = json.loads((path/'metrics.json').read_text())
        records.extend((path, fold) for fold in metrics['folds'])
    records.sort(key=lambda pair: pair[1]['year'])
    source = Path(config['dataset'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != config['dataset_sha256']:
        raise ValueError('Refit comparison source hash differs from frozen data')
    series = load_csv(source,provider=config['provider'],price_column=config['price_column'])
    batch = make_windows(series,length=config['window_length'],stride=config['score_stride'])
    rows = []
    for (prior_path,prior),(current_path,current) in zip(records[:-1],records[1:]):
        masks = split_masks(batch.dates,batch.price_start,**current['boundaries'],strict=True)
        x = batch.samples[masks['validation']]
        for name in ('w2','shape_w2'):
            stem = 'model' if name=='w2' else name
            a = WassersteinKMeans.load(prior_path/'models'/str(prior['year'])/stem)
            b = WassersteinKMeans.load(current_path/'models'/str(current['year'])/stem)
            row = compare_refits(a,b,x,shape_only=name=='shape_w2')
            row.update(model=name,previous_fold=prior['year'],current_fold=current['year'],
                       scoring_start=str(batch.dates[masks['validation']][0].date()),
                       scoring_end=str(batch.dates[masks['validation']][-1].date()))
            rows.append(row)
    return rows


def acquisition_metadata(config):
    source = Path(config['dataset'])
    metadata = json.loads(source.with_suffix('.json').read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if metadata.get('symbol') != config['symbol'] or metadata.get('sha256') != digest or digest != config['dataset_sha256']:
        raise ValueError('Acquisition metadata does not match frozen snapshot')
    return metadata


def summarize_asset(symbol, development, holdout):
    paths = dict(development=Path(development),holdout=Path(holdout))
    configs, metrics, manifests = {}, {}, {}
    for stage,path in paths.items():
        verify_artifact(path)
        configs[stage]=json.loads((path/'config.json').read_text())
        metrics[stage]=json.loads((path/'metrics.json').read_text())
        manifests[stage]=json.loads((path/'manifest.json').read_text())
        if configs[stage].get('symbol') != symbol or configs[stage].get('stage') != stage:
            raise ValueError('Asset or stage does not match the saved artifact')
    validate_protocol({stage:{k:v for k,v in config.items() if k!='stage'} for stage,config in configs.items()})
    if configs['development']['dataset_sha256'] != configs['holdout']['dataset_sha256']:
        raise ValueError('Development and holdout must use the same frozen snapshot')
    latest=metrics['holdout']['folds'][-1]
    raw=latest['models']['w2']['strict']
    shape=latest['models']['shape_w2']['strict']
    stability=metrics['holdout']['stability']['models']
    histories=refit_history(development,holdout)
    shape_refits=[r for r in histories if r['model']=='shape_w2' and not r['both_single_state']]
    periods=[]
    for stage in ('development','holdout'):
        for fold in metrics[stage]['folds']:
            periods.append(dict(stage=stage,year=fold['year'],k=fold['k'],
                                raw_scale_fraction=fold['models']['w2']['strict']['centroid_decomposition']['scale'],
                                raw_vs_volatility_ari=fold['ari_against_raw_w2']['volatility'],
                                shape_occupancy=fold['models']['shape_w2']['strict']['occupancy'],
                                raw_occupancy=fold['models']['w2']['strict']['occupancy']))
    row=dict(symbol=symbol,acquisition=acquisition_metadata(configs['holdout']),k=latest['k'],strict_windows=raw['n'],
             training_start=manifests['holdout']['training_period'][0],
             raw_centroid_decomposition=raw['centroid_decomposition'],
             raw_vs_volatility_ari=latest['ari_against_raw_w2']['volatility'],
             raw_vs_rich_ari=latest['ari_against_raw_w2']['rich'],
             raw_vs_shape_ari=latest['ari_against_raw_w2']['shape_w2'],
             shape_occupancy=shape['occupancy'],raw_occupancy=raw['occupancy'],
             shape_seed_ari=stability['shape_w2']['seed']['ari_mean'],
             shape_bootstrap_ari=stability['shape_w2']['block_bootstrap']['ari_mean'],
             shape_bootstrap_range=stability['shape_w2']['block_bootstrap']['ari_range'],
             raw_bootstrap_ari=stability['w2']['block_bootstrap']['ari_mean'],
             volatility_bootstrap_ari=stability['volatility']['block_bootstrap']['ari_mean'],
             shape_refit_ari_median=float(np.median([r['ari'] for r in shape_refits])) if shape_refits else None,
             shape_informative_refits=len(shape_refits),refits=histories,periods=periods,
             overlap=metrics['holdout']['overlap'],manifests=manifests,
             runs={stage:path.name for stage,path in paths.items()})
    return row


def render_comparison(summary, output):
    """Render aggregate measurements; no fits or data downloads."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rows=summary['assets']
    table=pd.DataFrame([dict(asset=r['symbol'],K=r['k'],scale_share=r['raw_centroid_decomposition']['scale'],
                             raw_volatility_ARI=r['raw_vs_volatility_ari'],shape_bootstrap_ARI=r['shape_bootstrap_ari'],
                             shape_refit_median_ARI=r['shape_refit_ari_median']) for r in rows])
    table.to_csv(output/'comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    x=np.arange(len(rows))
    axes[0].bar(x,[r['raw_centroid_decomposition']['scale'] for r in rows],color='#2862a7')
    axes[0].set(xticks=x,xticklabels=[r['symbol'] for r in rows],ylim=(0,1.05),ylabel='Share of between-centroid squared W2',title='Raw regimes: scale contribution')
    axes[1].bar(x-.2,[r['shape_bootstrap_ari'] for r in rows],width=.4,label='Return-block refit mean',color='#278766')
    axes[1].bar(x+.2,[r['shape_refit_ari_median'] if r['shape_refit_ari_median'] is not None else np.nan for r in rows],width=.4,label='Consecutive refit median',color='#b95816')
    axes[1].set(xticks=x,xticklabels=[r['symbol'] for r in rows],ylim=(-.1,1.05),ylabel='ARI on validation windows',title='Shape-only stability')
    axes[1].legend(fontsize=8,loc='lower left')
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)
    encoded=base64.b64encode((output/'comparison.png').read_bytes()).decode()
    sections=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Cross-asset regime replication</title>',
              '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#17283d}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd;padding:8px;text-align:right}img{max-width:100%}pre{white-space:pre-wrap}</style>',
              '<h1>Cross-asset regime replication</h1><p>Independent frozen QQQ, TLT, GLD and HYG models; identical methodology, different inception histories. No pooling, forecasting or trading claim.</p>',
              '<p>Centroid geometry is learned from training; agreement is measured on the 2024–August 2026 strict holdout. Block and consecutive-refit stability use validation windows. These statistics have different meanings.</p>',
              table.to_html(index=False,float_format=lambda x:f'{x:.3f}',escape=True),
              f'<img alt="Raw scale contribution and shape-only validation stability by asset" src="data:image/png;base64,{encoded}">',
              '<p>Seed agreement alone is insufficient. Bootstrap ARI ranges, changing K, sparse occupancy and per-period refits appear below. ARI=1 when both models use one state is flagged and excluded from the headline refit median. The four ETFs are correlated and are not independent statistical replications. Snapshot revisions and unequal history lengths limit causal attribution to asset class.</p>']
    for row in rows:
        sections.extend([f'<h2>{html.escape(row["symbol"])}</h2>',
                         pd.DataFrame(row['periods']).to_html(index=False,escape=True),
                         pd.DataFrame(row['refits']).to_html(index=False,escape=True),
                         '<details><summary>Complete aggregate diagnostics and provenance</summary><pre>'+html.escape(json.dumps(row,indent=2))+'</pre></details>'])
    sections.append('</html>')
    (output/'report.html').write_text('\n'.join(sections))
    write_json(output/'summary.json',summary)
    return output/'report.html'


def run_cross_asset(config_path, *, output_root='artifacts', stage='development'):
    """Execute all development studies before opening any new holdout."""
    specification=yaml.safe_load(Path(config_path).read_text())
    configs={symbol:yaml.safe_load(Path(path).read_text()) for symbol,path in specification['assets'].items()}
    reference_config=specification.get('reference_config')
    if reference_config:
        reference=yaml.safe_load(Path(reference_config).read_text())
        validate_protocol(dict(configs,SPY_reference=reference))
    else:
        validate_protocol(configs)
    if stage not in ('all','development','holdout'):
        raise ValueError('Invalid cross-asset stage')
    index_path=Path(output_root)/'cross_asset_runs.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else {}
    stages=('development','holdout') if stage=='all' else (stage,)
    for phase in stages:
        if phase == 'holdout':
            for symbol,expected in configs.items():
                saved = index.get(symbol,{}).get('development')
                if saved is None:
                    raise ValueError(f'{symbol}: completed development is required before holdout')
                verify_artifact(saved)
                prior=json.loads((Path(saved)/'config.json').read_text())
                if prior != dict(expected,stage='development'):
                    raise ValueError(f'{symbol}: development configuration differs from frozen holdout specification')
        for symbol,path in specification['assets'].items():
            print(f'{symbol}: {phase}',flush=True)
            folder=run(path,output_root=output_root,stage=phase)
            report(folder)
            index.setdefault(symbol,{})[phase]=str(folder)
            write_json(index_path,index)
    if stage=='development':
        return index_path
    selected={symbol:index[symbol] for symbol in configs}
    if any('development' not in paths or 'holdout' not in paths for paths in selected.values()):
        raise ValueError('Both development and holdout artifacts are required for comparison')
    rows=[summarize_asset(symbol,paths['development'],paths['holdout']) for symbol,paths in selected.items()]
    identity=artifact_id(configs,{r['symbol']:dict(runs=r['runs'],acquisition=r['acquisition']) for r in rows})
    reference_path=specification.get('reference_holdout')
    reference=json.loads(Path(reference_path).read_text()) if reference_path else None
    summary=dict(study_id=identity,assets=rows,protocol=specification,spy_reference=reference,
                 interpretation='Descriptive replication; no predictive-value test')
    return render_comparison(summary,Path(output_root)/f'cross-asset-{identity}')
