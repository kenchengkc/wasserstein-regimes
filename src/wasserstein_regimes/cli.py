# SPDX-License-Identifier: GPL-3.0-only
"""Local research commands."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(prog='regimes')
    sub=parser.add_subparsers(dest='command',required=True)
    runner=sub.add_parser('run',help='Run the frozen local-data research pipeline')
    runner.add_argument('--config',required=True)
    runner.add_argument('--stage',choices=['development','holdout'],default='development')
    runner.add_argument('--output-root',default='artifacts')
    reporter=sub.add_parser('report',help='Regenerate report solely from saved artifacts')
    reporter.add_argument('--run',required=True,help='Run ID under artifacts/ or path to run directory')
    synth=sub.add_parser('synthetic',help='Run synthetic controls')
    synth.add_argument('--config',required=True)
    synth.add_argument('--output',default='artifacts/synthetic-standalone.json')
    cross=sub.add_parser('cross-asset',help='Run the frozen independent-asset replication')
    cross.add_argument('--config',required=True)
    cross.add_argument('--stage',choices=['all','development','holdout'],default='development')
    cross.add_argument('--output-root',default='artifacts')
    joint=sub.add_parser('joint-study',help='Run the frozen exploratory synchronized panel study')
    joint.add_argument('--config',required=True)
    joint.add_argument('--stage',choices=['development','assessment'],default='development')
    joint.add_argument('--development',help='Sealed development directory; required for assessment')
    joint.add_argument('--output-root',default='artifacts')
    args=parser.parse_args()
    if args.command=='run':
        from .experiments import run
        print(run(args.config,output_root=args.output_root,stage=args.stage))
    elif args.command=='cross-asset':
        from .cross_asset import run_cross_asset
        print(run_cross_asset(args.config,output_root=args.output_root,stage=args.stage))
    elif args.command=='joint-study':
        from .joint_market import run_joint_market
        print(run_joint_market(args.config,output_root=args.output_root,stage=args.stage,development=args.development))
    elif args.command=='report':
        from .reporting import report
        path=Path(args.run)
        print(report(path if path.is_dir() else Path('artifacts')/path))
    else:
        import yaml
        from .synthetic import run_synthetic
        from .experiments import write_json
        result=run_synthetic(yaml.safe_load(Path(args.config).read_text()))
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        write_json(args.output,result)
        print(args.output)


if __name__=='__main__':
    main()
