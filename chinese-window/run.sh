#!/usr/bin/env bash

set -e

output_file=$(mktemp output/outputXXX.svg)

./main.py $output_file

eog $output_file
