# SPDX-License-Identifier: GPL-3.0-only
"""Standalone entry for the packaged transport benchmark."""
import json
from wasserstein_regimes.benchmark import run_benchmark

if __name__ == '__main__':
    print(json.dumps(run_benchmark(), indent=2))
