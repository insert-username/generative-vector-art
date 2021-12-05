# Maze Generator

Uses randomized depth first with various biasing methods to create mazes!

## Example usage:

```
./main.py --rows=16 --columns=16 --svg_height=400 --svg_width=400 --border_size=20 --cell_line_width=4 output/maze1
```

`rows/columns`: Maze cell dimensions
`svg_width/height`: "size" of the generated SVG, the exact value doesn't matter (within reason). What changes the appearance is the relative sizes of the border etc.
`border_size`: Radius of the cut-out border surrounding the coaster
`cell_line_width`: Thickness of the maze "Walls", useful if you want to fill them for engraving.

## Output
Output is 3 files:

- `filename.svg`: Set of filled svg rectangles representing the maze walls.
- `filename-bbox.svg`: Outline that can be used to cut out the coaster.
- `filename-path.svg`: Optional outline of the maze walls. Use this if you want to get crisp edges after rastering.

## Using Images for biasing
You can do this through the command line, but it's also pretty easy to do it
in python (either by modifying main.py or writing your own script). There are
three types used in the generation process:

- `neighbor_selector`: picks from the available neighbors (an available one must always be selected, but it can be biased to prefer a given direction.
- `bias_provider`: Use with DirectionalBiasedNeighborSelector. The bias provider assigns each cell a "rank", which the NS then uses to pick the preferred neighbor. One example is ConstDirectionalBiasedProvider. It uses a value between 0-1 to rank cells. Vertical cells are ranked rng(0, 1.0 - value), horizontal rng(0, value). If value is 0.5 all cells have the same probability of receiving a given rank, but if value is 0 or 1, the preferred neighbor becomes biased to vertical or horizontal, so you get a maze with e.g. more horizontal paths.
    - ImageDirectionalBWBiasProvider: use an image to bias to horizontal/vertical
    - ImageDirectionalBiasProvider: brightness of an image is mapped from 0-1 to 0-2pi and used as phase to prefer a given orientation.
- ImageDelegatingNS: Allows you to use two different neighbor selectors, and vary the bias towards one or the other using an image (image is converted to greyscale). For example you could have a maze which prefers to be horizontal at the bottom and vertical at the top, with a transition between the two according to a gradient image.
- CurlingNS: A neighbor selector that prefers to draw spirals of a given size.

