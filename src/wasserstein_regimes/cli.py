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
    validation=sub.add_parser('joint-validate',help='Run joint regime robustness checks on validation only')
    validation.add_argument('--config',required=True)
    validation.add_argument('--development',required=True,help='Sealed joint-market development directory')
    validation.add_argument('--output-root',default='artifacts')
    freeze=sub.add_parser('freeze-joint',help='Export a verified frozen joint model')
    freeze.add_argument('--development',required=True)
    freeze.add_argument('--model',choices=['scaled_joint','raw_joint'],default='scaled_joint')
    freeze.add_argument('--output-root',default='artifacts/bundles')
    score=sub.add_parser('score-joint',help='Score snapshots using a frozen joint contract')
    score.add_argument('--bundle',required=True)
    score.add_argument('--config',required=True)
    score.add_argument('--output-root',default='artifacts/jobs')
    score.add_argument('--blas-threads',type=int,default=1)
    score.add_argument('--timeout',type=float,default=3600)
    jobs=sub.add_parser('run-jobs',help='Run or resume a bounded local experiment batch')
    jobs.add_argument('--config',required=True)
    jobs.add_argument('--output-root',default='artifacts/jobs')
    jobs.add_argument('--workers',type=int,default=2)
    jobs.add_argument('--blas-threads',type=int,default=1)
    jobs.add_argument('--timeout',type=float,default=3600)
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
    elif args.command=='joint-validate':
        from .joint_validation import run_joint_validation
        print(run_joint_validation(args.config,development=args.development,output_root=args.output_root))
    elif args.command=='freeze-joint':
        from .frozen import freeze_joint
        print(freeze_joint(args.development,model=args.model,output_root=args.output_root))
    elif args.command=='score-joint':
        from .jobs import run_jobs
        print(run_jobs([dict(name='frozen-score',type='score',bundle=args.bundle,config=args.config)],
            output_root=args.output_root,workers=1,blas_threads=args.blas_threads,timeout=args.timeout))
    elif args.command=='run-jobs':
        from .jobs import run_job_file
        print(run_job_file(args.config,output_root=args.output_root,workers=args.workers,
            blas_threads=args.blas_threads,timeout=args.timeout))
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
