#!/usr/bin/env python3

import shapely as sh
import shapely.ops as sho

import numpy as np

import cairo

import argparse
import random

def get_simple_line_angle(line):
    if len(line.coords) != 2:
        raise ValueError("Line must have length 2")

    coord0 = line.coords[0]
    coord1 = line.coords[1]

    return np.arctan2(coord1[0] - coord0[0], coord1[1] - coord0[1])

def get_distributed_values(minval, maxval, count, closest_tolerance=0.2):
    values = []

    while len(values) < count:
        candidate = random.uniform(minval, maxval)

        if not any(abs(v - candidate) < closest_tolerance for v in values):
            values.append(candidate)

    return values

def get_circles(svg_width, svg_height, count):
    bounds = sh.geometry.box(0, 0, svg_width, svg_height)

    result = []
    for r in get_distributed_values(0.2 * svg_width, 0.9 * svg_width, count):
        circle = sh.geometry.Point(0, 0).buffer(r, resolution=100)

        circle_line = []

        for i in range(0, len(circle.boundary.coords)):
            if bounds.contains(sh.geometry.Point(circle.boundary.coords[i])):
                circle_line.append(circle.boundary.coords[i])

        result.append(sh.geometry.LineString(circle_line))

    return result

def get_horiz_lines(svg_width, svg_height, count):

    return [ sh.geometry.LineString([ (0, ycoord), (svg_width, ycoord)  ]) \
            for ycoord in get_distributed_values(svg_height * 0.1, svg_height * 0.9, count) ]

def get_vert_lines(svg_width, svg_height, count):

    return [ sh.geometry.LineString([ (coord, 0), (coord, svg_height)  ]) \
            for coord in get_distributed_values(svg_width * 0.1, svg_width * 0.9, count) ]


def simplify_cross_step(line_collection):

    for i in range(0, len(line_collection)):
        lineA = line_collection[i]

        for j in range(0, len(line_collection)):
            if i == j:
                continue

            lineB = line_collection[j]

            if lineA.crosses(lineB):
                min_index = min(i, j)
                max_index = max(i, j)
                del line_collection[max_index]
                del line_collection[min_index]

                for l in lineA.symmetric_difference(lineB).geoms:
                    if (l.type != "LineString"):
                        raise RuntimeError(f"Symmetric difference of {lineA} and {lineB} resulted in {l}, which is not of type LineString")

                    line_collection.append(l)

def simplify_cross(line_collection):

    while True:
        prev_length = len(line_collection)
        simplify_cross_step(line_collection)

        print(prev_length)

        if len(line_collection) == prev_length:
            # nothing was done
            return

def is_horiz_or_vert(line):
    return ((line.coords[0][0] == line.coords[1][0]) or (line.coords[0][1] == line.coords[1][1]))

def simplify_intersections_step(line_collection):
    intersections = {}

    for i in range(0, len(line_collection)):
        lineA = line_collection[i]
        if not is_horiz_or_vert(lineA):
            # do not consider non horizontal/vertical lines
            continue

        intersections[i] = []
        for j in range(0, len(line_collection)):
            if i == j:
                continue

            lineB = line_collection[j]

            if lineA.touches(lineB) and is_horiz_or_vert(lineB):
                intersections[i].append(j)

    for i in range(0, len(line_collection)):
        if i in intersections and len(intersections[i]) > 2:
            intersecting_lines = intersections[i]
            intersecting_lines.sort(key=lambda x: len(intersections[x]))

            minimum_connector = intersecting_lines[0]

            #connector should be attached to at least one other line
            if len(intersections[minimum_connector]) > 2:
                print(f"Removing element {minimum_connector}")
                del line_collection[minimum_connector]
                return


def simplify_intersections(line_collection):
    while True:
        init_len = len(line_collection)
        simplify_intersections_step(line_collection)
        if len(line_collection) == init_len:
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", metavar="OUTPUT")
    args = parser.parse_args()

    SVG_WIDTH = 400
    SVG_HEIGHT = 400
    surface = cairo.SVGSurface(args.output, SVG_WIDTH, SVG_HEIGHT)
    c = cairo.Context(surface)
    c.set_line_width(2)

    shapes = []

    shapes += get_circles(SVG_WIDTH, SVG_HEIGHT, 6)
    shapes += get_vert_lines(SVG_WIDTH, SVG_HEIGHT, 6)
    shapes += get_horiz_lines(SVG_WIDTH, SVG_HEIGHT, 6)
    shapes.append(sh.geometry.LineString([(0,0), (SVG_WIDTH, SVG_HEIGHT)]))

    for shape in shapes:
        print(shape)

    for s in shapes:
        if not (s.type == "LineString" or s.type == "Polygon"):
            raise RuntimeError(f"All shape types must be linestring, but encountered {s}")

    # end up with a collection of linestrings
    # linestring.crosses(other)

    lines = []
    for shape in shapes:
        if shape.type == "LineString":
            lines.append(shape)
        else:
            lines.append(shape.boundary)

    print(f"Splitting {len(lines)} linestrings into crossing lines")
    simplify_cross(lines)

    print(f"Removing horizontal or vertical lines with more than two intersection points.")
    simplify_intersections(lines)

    lines_next = []
    for line in lines:
        if random.uniform(0,1) > 0.5:
            lines_next.append(line)

    lines = lines_next

    print(f"Scaling everything/mirroring")
    lines_next = []
    for line in lines:
        center = (SVG_WIDTH/2, SVG_HEIGHT/2)

        line_bottom_left = \
            sh.affinity.scale(line, xfact=0.5, yfact=0.5, origin=(0, 0))
        line_bottom_left = \
            sh.affinity.translate(line_bottom_left, xoff=SVG_WIDTH/2, yoff=SVG_HEIGHT/2)

        line_bottom_right = \
            sh.affinity.scale(line_bottom_left, xfact=-1, yfact=1, origin=center)

        line_top_right = \
            sh.affinity.scale(line_bottom_right, xfact=1, yfact=-1, origin=center)

        line_top_left = \
            sh.affinity.scale(line_bottom_left, xfact=1, yfact=-1, origin=center)

        lines_next.append(line_bottom_left)
        lines_next.append(line_bottom_right)
        lines_next.append(line_top_left)
        lines_next.append(line_top_right)


    lines = lines_next

    print(f"Drawing {len(lines)} lines...")
    for line in lines:

        coords = [ coord for coord in line.coords ]

        c.move_to(coords[0][0], coords[0][1])
        for coord in coords:
            c.line_to(coord[0], coord[1])

        #c.set_source_rgba(random.uniform(0,0.5),random.uniform(0,0.5),random.uniform(0,0.5),1)
        c.stroke()

    print("Saving")

    surface.finish()

    print("Done")

