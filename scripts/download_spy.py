# SPDX-License-Identifier: GPL-3.0-only
"""Acquire a new ETF snapshot; never overwrite a frozen study input."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import yfinance as yf


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--symbol', default='SPY', choices=['SPY','QQQ','TLT','GLD','HYG'])
    parser.add_argument('--output',required=True,help='New CSV path; existing paths are rejected')
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists() or path.with_suffix('.json').exists():
        raise SystemExit('Refusing to overwrite an existing snapshot')
    data=yf.download(args.symbol,start='1993-01-01',end='2026-09-01',interval='1d',
                     auto_adjust=False,actions=True,repair=False,keepna=True,
                     threads=False,progress=False,multi_level_index=False,timeout=30)
    if data.empty:
        raise SystemExit('Provider returned no observations')
    path.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(path,index_label='Date',float_format='%.12g')
    metadata=dict(provider='Yahoo Finance via yfinance',yfinance_version=yf.__version__,symbol=args.symbol,
                  retrieved_at=datetime.now(timezone.utc).isoformat(),return_basis='adjusted_close',
                  point_in_time=False,rows=len(data),first_date=str(data.index[0].date()),
                  last_date=str(data.index[-1].date()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  auto_adjust=False,repair=False,keepna=True,redistribution='not bundled; personal research snapshot')
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(path)
    print('A later download may have a different hash because adjusted history can be revised.')


if __name__=='__main__':
    main()
