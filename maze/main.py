#!/usr/bin/env python3

import cairo

from PIL import Image, ImageOps

import math
import random
import argparse

MAZE_ROWS=20
MAZE_COLUMNS=20

# generation algorithm:
# start with all walls erected
# pick a starting location
#    until every cell has been visited
#    find a neighbor that has not been visited
#      push it to the stack & delete walls between
#    if all have been visited
#      pop from stack


class Cell:

    def __init__(self, row, column):
        self._row = row
        self._column = column

        self.walls = {
                "north": True,
                "south": True,
                "east": True,
                "west": True
            }

        self.neighbors = {
                "north": None,
                "south": None,
                "east": None,
                "west": None
            }

    def get_row(self):
        return self._row

    def get_column(self):
        return self._column

    def north(self):
        return self.neighbors["north"]

    def south(self):
        return self.neighbors["south"]

    def east(self):
        return self.neighbors["east"]

    def west(self):
        return self.neighbors["west"]

    def set_north(self, other):
        self.neighbors["north"] = other
        other.neighbors["south"] = self

    def set_south(self, other):
        self.neighbors["south"] = other
        other.neighbors["north"] = self

    def set_east(self, other):
        self.neighbors["east"] = other
        other.neighbors["west"] = self

    def set_west(self, other):
        self.neighbors["west"] = other
        other.neighbors["east"] = self

    def knock_walls(self, other):
        if (self.east() is not None and self.east() == other and other.west() == self):
            self.walls["east"] = False
            other.walls["west"] = False

        elif (self.west() is not None and self.west() == other and other.east() == self):
            self.walls["west"] = False
            other.walls["east"] = False

        elif (self.north() is not None and self.north() == other and other.south() == self):
            self.walls["north"] = False
            other.walls["south"] = False

        elif (self.south() is not None and self.south() == other and other.north() == self):
            self.walls["south"] = False
            other.walls["north"] = False
        else:
            my_neighbors = ""
            for direction, neighbor in self.neighbors.items():
                my_neighbors += f"\n{direction}: {neighbor}"

            other_neighbors = ""
            for direction, neighbor in other.neighbors.items():
                other_neighbors += f"\n{direction}: {neighbor}"

            raise RuntimeError(f"Cell is not a neighbor,\n\nmy {self} neighbors {my_neighbors},\n\ntheir {other} neighbors {other_neighbors}")

    def __str__(self):
        return f"({self._row}, {self._column} {id(self)})"

class Maze:

    def __init__(self, rows=10, columns=10):
        self.rows = rows
        self.columns = columns
        self.cells = []
        for row in range(0, rows):
            self.cells.append([ Cell(row, column) for column in range(0, columns) ])

        for row in range(0, rows):
            for column in range(0, columns):
                current_cell = self.cells[row][column]

                if row > 0:
                    current_cell.set_north(self.cells[row - 1][column])

                if column > 0:
                    current_cell.set_west(self.cells[row][column - 1])

                if row < rows - 1:
                    current_cell.set_south(self.cells[row + 1][column])

                if column < columns - 1:
                    current_cell.set_east(self.cells[row][column + 1])


    def get_element_count(self):
        return self.rows * self.columns

    def __str__(self):
        return f"{self.rows} x {self.columns}"

class Generator:

    def __init__(self, maze, bias_provider):
        self.cell_stack = []
        self.visited_cells = set()
        self.maze = maze
        self.bias_provider = bias_provider

    def step(self):

        if (len(self.visited_cells) == self.maze.get_element_count()):
            # algorithm has finished
            return True
        elif len(self.visited_cells) == 0:
            start_cell = self.maze.cells[0][0]
            self.visited_cells.add(start_cell)
            self.cell_stack.append(start_cell)
            return False

        current_cell = self.cell_stack[-1]

        unvisited_neighbors = self._get_unvisited_neighbors(current_cell)

        if (len(unvisited_neighbors) == 0):
            self.cell_stack.pop()
        else:
            selected_neighbor = self._select_neighbor(current_cell, unvisited_neighbors)

            current_cell.knock_walls(selected_neighbor)

            self.cell_stack.append(selected_neighbor)
            self.visited_cells.add(selected_neighbor)


        return False

    def _get_unvisited_neighbors(self, cell):
        result = []

        for name, neighbor in cell.neighbors.items():
            if neighbor is not None and not neighbor in self.visited_cells:
                result.append(neighbor)


        return result

    def _select_neighbor(self, cell, neighbors):
        bias = self.bias_provider.get_cell_bias(cell)

        selected_neighbor = random.choice(neighbors)

        if bias == 0 or random.uniform(0.0, 1.0) > abs(bias):
            return selected_neighbor

        sub_neighbor_pool = []
        if bias < 0:
            sub_neighbor_pool = [n for n in neighbors if n.get_row() == cell.get_row()]
        elif bias > 0:
            sub_neighbor_pool = [n for n in neighbors if n.get_column() == cell.get_column()]

        if len(sub_neighbor_pool) == 0:
            # no ideal choices available, so select the original
            return selected_neighbor
        else:
            return random.choice(sub_neighbor_pool)

class MazeCoordTranslator:

    def __init__(self, maze, line_width, svg_x0, svg_y0, svg_width, svg_height):
        self.line_width = line_width
        self.svg_x0 = svg_x0
        self.svg_y0 = svg_y0
        self.svg_width = svg_width
        self.svg_height = svg_height

        self.row_height = svg_height / maze.rows
        self.column_width = svg_width / maze.columns

        self.wall_length = self.column_width - self.line_width
        self.wall_height = self.row_height - self.line_width

    # returns the cell boundary coords
    def get_cell(self, row, column):
        x0 = self.svg_x0 + self.column_width * column
        x1 = self.svg_x0 + self.column_width * (column + 1)

        y0 = self.svg_y0 + self.row_height * row
        y1 = self.svg_y0 + self.row_height * (row + 1)

        return (x0, y0, x1, y1)

    def get_cell_interior(self, row, column):
        x0, y0, x1, y1 = self.get_cell(row, column)

        hlw = self.line_width / 2

        return (x0 + hlw, y0 + hlw, x1 - hlw, y1 - hlw)

    def get_cell_exterior(self, row, column):
        x0, y0, x1, y1 = self.get_cell(row, column)

        hlw = self.line_width / 2

        return (x0 - hlw, y0 - hlw, x1 + hlw, y1 + hlw)

    def get_north_wall_rect(self, row, column):
        x0_int, y0_int, x1_int, y1_int = self.get_cell_interior(row, column)
        x0_ext, y0_ext, x1_ext, y1_ext = self.get_cell_exterior(row, column)

        return (x0_int, y0_ext, self.wall_length, self.line_width)

    def get_south_wall_rect(self, row, column):
        x0_int, y0_int, x1_int, y1_int = self.get_cell_interior(row, column)
        x0_ext, y0_ext, x1_ext, y1_ext = self.get_cell_exterior(row, column)

        return (x0_int, y1_int, self.wall_length, self.line_width)

    def get_east_wall_rect(self, row, column):
        x0_int, y0_int, x1_int, y1_int = self.get_cell_interior(row, column)
        x0_ext, y0_ext, x1_ext, y1_ext = self.get_cell_exterior(row, column)

        return (x1_int, y0_int, self.line_width, self.wall_height)

    def get_west_wall_rect(self, row, column):
        x0_int, y0_int, x1_int, y1_int = self.get_cell_interior(row, column)
        x0_ext, y0_ext, x1_ext, y1_ext = self.get_cell_exterior(row, column)

        return (x0_ext, y0_int, self.line_width, self.wall_height)

    def get_bottom_right_corner_rect(self, row, column):
        x0_int, y0_int, x1_int, y1_int = self.get_cell_interior(row, column)
        x0_ext, y0_ext, x1_ext, y1_ext = self.get_cell_exterior(row, column)
        return (x1_int, y1_int, self.line_width, self.line_width)

    @staticmethod
    def rect(c, rect):
        c.rectangle(rect[0], rect[1], rect[2], rect[3])

def create_coaster(maze, output_file, svg_width=400, svg_height=400, border_width=None, line_width=1):
    border_width = border_width if border_width is not None else svg_width / 8
    arc_r = border_width

    translator = MazeCoordTranslator(maze,
                                     line_width,
                                     border_width,
                                     border_width,
                                     svg_width - border_width * 2,
                                     svg_height - border_width * 2)

    # surface representing the "fill" of the maze
    surface = cairo.SVGSurface(output_file + ".svg", svg_width, svg_height)
    c = cairo.Context(surface)
    render_maze(maze,
            output_file,
            c,
            translator)
    surface.finish()

    surface = cairo.SVGSurface(output_file + "-path.svg", svg_width, svg_height)
    c = cairo.Context(surface)
    render_interior_path(maze, c, translator)
    surface.finish()

    # create the bounding box
    surface = cairo.SVGSurface(output_file + "-bbox.svg", svg_width, svg_height)
    c = cairo.Context(surface)
    c.move_to(border_width, 0)
    c.line_to(svg_width - border_width, 0)
    c.arc(svg_width - border_width, border_width, arc_r, - math.pi / 2, 0)
    c.line_to(svg_width, svg_height - border_width)
    c.arc(svg_width - border_width, svg_height - border_width, arc_r, 0, math.pi / 2)
    c.line_to(border_width, svg_height)
    c.arc(border_width, svg_height - border_width, arc_r, math.pi / 2, - math.pi)
    c.line_to(0, border_width)
    c.arc(border_width, border_width, arc_r, -math.pi, -math.pi / 2)
    c.close_path()

    c.set_line_width(line_width)
    c.stroke()
    surface.finish()


def render_interior_path(maze, c, translator):
    c.set_source_rgb(0, 0, 0)

    # draw the bounding box (all exterior walls)
    top_left = translator.get_cell_exterior(0, 0)
    bottom_right = translator.get_cell_exterior(maze.rows - 1, maze.columns - 1)

    c.rectangle(top_left[0], top_left[1], bottom_right[2] - top_left[0], bottom_right[3] - top_left[1])
    c.stroke()

    for row in range(-1, maze.rows):
        for column in range(-1, maze.columns):

            br = translator.get_bottom_right_corner_rect(row, column)

            # if cell below does not have east wall, render bottom of rectangle
            if row < maze.rows -1 and not maze.cells[row + 1][column].walls["east"]:
                c.move_to(br[0],         br[1] + br[3])
                c.line_to(br[0] + br[2], br[1] + br[3])
                c.stroke()

            if column < maze.columns -1 and not maze.cells[row][column + 1].walls["south"]:
                c.move_to(br[0] + br[2], br[1])
                c.line_to(br[0] + br[2], br[1] + br[3])
                c.stroke()

            if row < 0 or column < 0:
                continue

            if not maze.cells[row][column].walls["east"]:
                c.move_to(br[0],         br[1])
                c.line_to(br[0] + br[2], br[1])
                c.stroke()

            if not maze.cells[row][column].walls["south"]:
                c.move_to(br[0], br[1])
                c.line_to(br[0], br[1] + br[3])
                c.stroke()

            has_north = maze.cells[row][column].walls["north"]
            has_south = maze.cells[row][column].walls["south"]
            has_east = maze.cells[row][column].walls["east"]
            has_west = maze.cells[row][column].walls["west"]

            x0_int, y0_int, x1_int, y1_int = translator.get_cell_interior(row, column)

            if has_north:
                c.move_to(x0_int, y0_int)
                c.line_to(x1_int, y0_int)
                c.stroke()

            if has_south:
                c.move_to(x0_int, y1_int)
                c.line_to(x1_int, y1_int)
                c.stroke()

            if has_east:
                c.move_to(x1_int, y0_int)
                c.line_to(x1_int, y1_int)
                c.stroke()

            if has_west:
                c.move_to(x0_int, y0_int)
                c.line_to(x0_int, y1_int)
                c.stroke()


def render_maze(maze, output_file, c, translator):
    c.set_source_rgb(0, 0, 0)

    for row in range(-1, maze.rows):
        for column in range(-1, maze.columns):

            # fill the bottom right corner
            MazeCoordTranslator.rect(c, translator.get_bottom_right_corner_rect(row, column))
            c.fill()

            if row < 0 or column < 0:
                continue

            has_north = maze.cells[row][column].walls["north"]
            has_south = maze.cells[row][column].walls["south"]
            has_east = maze.cells[row][column].walls["east"]
            has_west = maze.cells[row][column].walls["west"]

            if has_north:
                MazeCoordTranslator.rect(c, translator.get_north_wall_rect(row, column))
                c.fill()

            if has_west:
                MazeCoordTranslator.rect(c, translator.get_west_wall_rect(row, column))
                c.fill()

            if (row == maze.rows - 1):
                if has_south:
                    MazeCoordTranslator.rect(c, translator.get_south_wall_rect(row, column))
                    c.fill()

            if column == maze.columns - 1:
                if has_east:
                    MazeCoordTranslator.rect(c, translator.get_east_wall_rect(row, column))
                    c.fill()

class ImageBiasProvider:

    def __init__(self, source_image, maze):
        self.source_image = ImageOps.grayscale(source_image.resize((maze.rows, maze.columns)))

    def get_cell_bias(self, cell):
        pixel = self.source_image.getpixel((cell.get_column(), cell.get_row()))

        return (pixel / 255) * 2 - 1

class ConstBiasProvider:

    def __init__(self, value):
        self.value = value

    def get_cell_bias(self, cell):
        return self.value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate laser cutter maze templates.")
    parser.add_argument("--rows", metavar="ROWS", type=int, default=32)
    parser.add_argument("--columns", metavar="COLUMNS", type=int, default=32)
    parser.add_argument("--svg_height", type=int, default=400)
    parser.add_argument("--svg_width", type=int, default=400)
    parser.add_argument("--border_size", type=float, default=80)
    parser.add_argument("--cell_line_width", type=float, default=3)
    parser.add_argument("--bias_image", nargs='?', type=str, default=None)
    parser.add_argument("output_file")

    args = parser.parse_args()

    maze = Maze(args.rows, args.columns)

    bias_provider = ConstBiasProvider(0)
    if args.bias_image is not None:
        image = Image.open(args.bias_image)
        bias_provider = ImageBiasProvider(image, maze)

    gen = Generator(maze, bias_provider)

    image_index = 0
    while not gen.step():
        image_index += 1

    create_coaster(maze, args.output_file, args.svg_width, args.svg_height, args.border_size, args.cell_line_width)

