#!/bin/bash
# Convenient wrapper script for running the example test suite

cd "$(dirname "$0")/examples" || exit 1
python3 test_all_examples.py "$@"

