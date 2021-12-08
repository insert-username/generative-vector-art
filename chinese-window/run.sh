#!/usr/bin/env bash

set -e

rm output/*.svg

for i in {1..10}; do

    output_file=$(mktemp output/outputXXX.svg)

    ./main.py $output_file

done

#eog $output_file
