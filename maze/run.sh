#!/usr/bin/env bash

set -e;

#./main.py --rows=$1 --columns=$1 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=$2 --bias_image=images/grad.png output/maze1

./main.py --rows=64 --columns=64 --svg_height=400 --svg_width=400 --border_size=20 --cell_line_width=4 output/maze1
#./main.py --rows=32 --columns=32 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=4 output/maze2
#./main.py --rows=64 --columns=64 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=2 output/maze3
#./main.py --rows=256 --columns=256 --svg_height=400 --svg_width=400 --border_size=10 --cell_line_width=1 output/maze4


#inkscape --actions="select:surface1; verb:SelectionUnGroup; EditSelectAll; SelectionUnion; FileSave; FileClose;" --batch-process output.svg;
