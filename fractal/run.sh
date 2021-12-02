#!/usr/bin/env bash

set -x

./main.py --branch="1:0.5:90" --branch="1:0.5:-90" --end_depth=8 output/output.svg && inkview output/output.svg

