#!/usr/bin/env bash

set -e;

./main.py --rows=16 --columns=16 --svg_height=400 --svg_width=400 --border_size=20 --cell_line_width=4 output/maze1
./main.py --rows=32 --columns=32 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=4 output/maze2
./main.py --rows=64 --columns=64 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=2 output/maze3

inkview output/maze1.svg &
inkview output/maze2.svg &
inkview output/maze3.svg &

