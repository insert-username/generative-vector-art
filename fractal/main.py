#!/usr/bin/env python3

import cairo
import numpy as np
import math
import random
import argparse

def max_brightness(image, x0, y0, width, height):
    brightness = 0

    for x in range(x0, x0 + width):
        for y in range(y0, y0 + height):
            brightness = max(brightness, image.getpixel((x, y)))

    return brightness


class FractalLine:

    class BranchSpecifier:

        def __init__(self, start_fraction, length_fraction, transform_matrix):
            self.start_fraction = start_fraction
            self.length_fraction = length_fraction
            self.transform_matrix = transform_matrix

        @staticmethod
        def parse(string_val):
            args = string_val.split(":")

            if len(args) != 3:
                raise ValueError("Expect 3 arguments!")

            return FractalLine.BranchSpecifier(
                    float(args[0]),
                    float(args[1]),
                    FractalLine.rot_mat(math.radians(float(args[2]))))


    def __init__(self, p0, p1, branch_specifiers):
        self.p0 = p0
        self.p1 = p1
        self.branch_specifiers = branch_specifiers

    def get_sub_elements(self):
        # create single element, with p0 midway between p0->p1

        p0_to_p1 = np.subtract(self.p1, self.p0)

        result = []

        for cs in self.branch_specifiers:
            p0_new = np.add(self.p0, np.dot(cs.start_fraction, p0_to_p1))
            p1_new = np.add(p0_new, np.dot(cs.length_fraction, p0_to_p1))

            p1_new = np.subtract(p1_new, p0_new)
            p1_new = np.matmul(cs.transform_matrix, p1_new)
            p1_new = np.add(p1_new, p0_new)

            result.append(FractalLine(p0_new, p1_new, self.branch_specifiers))

        return result


    @staticmethod
    def rot_mat(theta):
        return [
                [ math.cos(theta), -math.sin(theta) ],
                [ math.sin(theta),  math.cos(theta) ]
            ]

    def render(self, c):
        c.move_to(self.p0[0], self.p0[1])
        c.line_to(self.p1[0], self.p1[1])
        c.stroke()


def draw_rect(source_image, c, x0, y0, width, height, depth, max_depth=None):
    if max_depth is None:
        max_depth = depth

    if depth < 0:
        return

    penetration_ratio = (max_depth - depth) / max_depth

    if max_brightness(image, x0, y0, width, height) < 150 * penetration_ratio:
        return

    c.rectangle(x0, y0, width, height)
    c.stroke()

    half_width = int(width / 2)
    half_height = int(height / 2)

    draw_rect(source_image, c, x0, y0, half_width, half_height, depth - 1, max_depth)
    draw_rect(source_image, c, x0 + half_width, y0, half_width, half_height, depth - 1, max_depth)
    draw_rect(source_image, c, x0, y0 + half_width, half_width, half_height, depth - 1, max_depth)
    draw_rect(source_image, c, x0 + half_width, y0 + half_width, half_width, half_height, depth - 1, max_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate svg fractals")
    parser.add_argument("--branch", metavar="BRANCH", type=str, nargs="+", action="append")
    parser.add_argument("depth", metavar="DEPTH", type=int)
    parser.add_argument("output_file", metavar="OUTPUT_FILE", type=str)

    args = parser.parse_args()

    branches = args.branch

    if len(branches) == 0:
        raise ValueError("Must specify at least one branch")

    width = 400
    height = 400

    depth = 8

    surface = cairo.SVGSurface(args.output_file, width, height)
    c = cairo.Context(surface)
    c.set_line_width(1)

    fractal_line = FractalLine(
            [ width / 2, height ],
            [ width / 2, height / 2],
            [ FractalLine.BranchSpecifier.parse(b[0]) for b in branches ])

    to_render = [fractal_line]
    depth = 0
    while depth < args.depth:
        next_to_render = []

        for f in to_render:
            f.render(c)
            next_to_render += f.get_sub_elements()

        to_render = next_to_render

        depth += 1

    surface.finish()
