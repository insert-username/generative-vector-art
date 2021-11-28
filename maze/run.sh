#!/usr/bin/env bash

set -e;

./main.py --rows=16 --columns=16 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=3 output/maze1
./main.py --rows=16 --columns=16 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=3 output/maze2
./main.py --rows=16 --columns=16 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=3 output/maze3




#inkscape --actions="select:surface1; verb:SelectionUnGroup; EditSelectAll; SelectionUnion; FileSave; FileClose;" --batch-process output.svg;
