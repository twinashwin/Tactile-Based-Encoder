#!/usr/bin/env bash
# Entry point the watchdog relaunches after a power cut. Runs the leaky folds
# first (nearly done), then the grouped version of the same combination.
# Both chains are resumable and skip cached feature sets.
set -u
cd "$(dirname "$0")" || exit 1
bash ftl_all_folds.sh
bash grouped_all_folds.sh
