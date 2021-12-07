#!/usr/bin/env python3

import shapely as sh
import shapely.ops as sho

import networkx as nx

import numpy as np

import math

import cairo

import argparse
import random

def get_simple_line_angle(line):
    if len(line.coords) != 2:
        raise ValueError("Line must have length 2")

    coord0 = line.coords[0]
    coord1 = line.coords[1]

    return np.arctan2(coord1[0] - coord0[0], coord1[1] - coord0[1])




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

# returns true if the specified lineA
# is able to cut lineB into more than
# one linestring
def will_cut(lineA, lineB):
    return (lineA.touches(lineB) or lineA.crosses(lineB)) and lineB.difference(lineA).type == "MultiLineString"

def will_any_cut(cutter_collection, cutee_collection):
    for cutter in cutter_collection:
        for cutee in cutee_collection:
            if will_cut(cutter, cutee):
                return True

    return False

def cut_against(line_to_cut, lines):

    # first establish which lines intersect
    intersecting_lines = [ l for l in lines if (l.touches(line_to_cut) or l.crosses(line_to_cut)) and line_to_cut.difference(l).type == "MultiLineString" ]

    #print(f"Line {line_to_cut} has {len(intersecting_lines)} intersecting lines,")

    result = [ line_to_cut ]
    has_cut = True
    while will_any_cut(intersecting_lines, result):
        print(len(result))
        for i in range(0, len(result)):
            sub_line = result[i]
            sub_intersecting = [ (i, l) for i, l in enumerate(intersecting_lines) if will_cut(l, sub_line) ]

            if len(sub_intersecting) == 0:
                continue
            else:
                diff = sub_line.difference(sub_intersecting[0][1]).geoms
                #print(f"Cut subline { sub_line } into:")
                #for l in diff:
                #    print(f"    {l}")
                #print(f"Via: {sub_intersecting[0][1]}")

                del result[i]
                result += diff

                break

    #print(f"{line_to_cut} became {len(result)} separate segments.")
    return result


def split_intersections(lines):
    new_lines = []

    for line in lines:
        new_lines += cut_against(line, [ l for l in lines if l != line ])

    # possible some lines are multilineStrings
    for line in new_lines:
        if not line.type == "LineString":
            raise RuntimeError(f"Result of line split was not type LineString: {line}")

    print(f"Split {len(lines)} original lines into {len(new_lines)} split lines:")
    print()

    return new_lines

def is_horiz_or_vert(line):
    return ((line.coords[0][0] == line.coords[1][0]) or (line.coords[0][1] == line.coords[1][1]))

def get_graph_representation(line_collection):
    graph = nx.Graph()

    # each line corresponds to a graph edge
    # each node is a line endpoint.

    def append_node(node):
        for existing_node in graph.nodes:
            dist = math.hypot(node[0] - existing_node[0], node[1] - existing_node[1])

            #tolerance seems to be needed due to small floating point errors
            if dist < 0.1:
                return existing_node

        graph.add_node(node)
        return node

    max_line_length = max(l.length for l in line_collection)
    min_line_length = min(l.length for l in line_collection)

    for i in range(0, len(line_collection)):
        line = line_collection[i]

        n0 = append_node(line.coords[0])
        n1 = append_node(line.coords[-1])

        weight = line.length

        #weight = random.uniform(min_line_length, max_line_length)

        if line.length != 0:
            # todo: there should not be any zero length lines!
            graph.add_edge(n0, n1, weight=weight, line=line)

    return graph


def prune_dingleberries(graph):
    # remove leaf edges with a length below a given threshold

    leaf_nodes = [ node for node in graph.nodes if graph.degree(node) == 1 ]
    leaf_edges = [ edge for edge in graph.edges if graph.degree(edge[0]) == 1 or graph.degree(edge[1]) == 1 ]

    removed_count = 0
    for leaf_edge in leaf_edges:
        edge_len = graph.edges[leaf_edge]["line"].length
        if edge_len < 10:
            removed_count += 1
            graph.remove_edge(leaf_edge[0],  leaf_edge[1])

    print(f"Removed {removed_count} dingleberries")

    return graph

class RandomUtils:

    @staticmethod
    def get_distributed_values(minval, maxval, count, closest_tolerance=0.2):
        values = []

        if (abs(maxval - minval) / closest_tolerance < count):
            raise ValueError("Number of requested values within tolerance is impossible.")

        while len(values) < count:
            candidate = random.uniform(minval, maxval)

            if not any(abs(v - candidate) < closest_tolerance for v in values):
                values.append(candidate)

        return values

class GeneratorExtent:

    def __init__(self, xMin, yMin, xMax, yMax):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.width = xMax - xMin
        self.height = yMax - yMin
        self.bounds = sh.geometry.box(self.xMin, self.yMin, self.xMax, self.yMax)
        self.center = (self.xMin + self.width / 2, self.yMin + self.height / 2)

    def random_point(self):
        x = random.uniform(self.xMin, self.xMax)
        y = random.uniform(self.yMin, self.yMax)

        return (x, y)

    def clip_linestring(self, linestring):
        if linestring.type != "LineString":
            raise ValueError(f"Clipped type must be linestring, instead was {linestring.type}.")

        result = linestring.intersection(self.bounds)
        print("INTERSECTING:")
        print(linestring)
        print("WITH")
        print(self.bounds)
        print("RESULT")
        print(result)

        if result.type == "MultiLineString":
            return [ l for l in result.geoms if not l.is_empty ]
        elif not result.empty:
            return [ result ]
        else:
            return []

class RandomLineGenerator:

    def __init__(self, extent):
        self.extent = extent

    def get_linestrings(self):
        extent = self.extent

        result = []

        result += self._get_circles(3)
        result += self._get_vert_lines(5, 50)
        result += self._get_horiz_lines(5, 50)

        box = sh.geometry.box(extent.xMin + extent.width * 0.3, extent.yMin + extent.height * 0.3, extent.xMin + extent.width * 0.7, extent.yMin + extent.height * 0.7)

        result.append(sh.affinity.rotate(box, 45).boundary)

        result.append(sh.geometry.LineString([  ( \
                extent.xMin + random.uniform(0, extent.width),) * 2, \
                (extent.xMax, extent.yMax)]))
        result.append(sh.geometry.LineString([  (extent.xMin + random.uniform(0, extent.width),0),       (extent.xMax, extent.yMin)   ]  ))
        result.append(sh.geometry.LineString([  (extent.xMin,  random.uniform(0, extent.width)),         (extent.yMin, extent.yMax)  ] ))

        return result

    def _get_circles(self, count):
        result = []
        for r in RandomUtils.get_distributed_values(extent.xMin + 0.2 * extent.width, extent.xMin + 0.9 * extent.width, count, closest_tolerance=0.05 * extent.width):
            resolution = 100
            points = []
            for i in range(0, resolution + 1):
                theta = math.pi * i / resolution
                x = r * math.cos(theta)
                y = r * math.sin(theta)

                points.append((x,y))

            result.append(sh.geometry.LineString(points))

        return result

    def _get_horiz_lines(self, count, tolerance):

        return [ sh.geometry.LineString([ (extent.xMin, coord), (extent.xMax, coord)  ]) \
                for coord in RandomUtils.get_distributed_values(extent.yMin + tolerance / 2, extent.yMax - tolerance / 2, count, tolerance) ]

    def _get_vert_lines(self, count, tolerance):

        return [ sh.geometry.LineString([ (coord, extent.yMin), (coord, extent.yMax)  ]) \
                for coord in RandomUtils.get_distributed_values(extent.xMin + tolerance / 2, extent.xMax - tolerance / 2, count, tolerance) ]

class CirclesShapeGenerator:

    def __init__(self, extent):
        self.extent = extent

    def get_linestrings(self):
        result = []

        for i in range(0,15):
            circle_linestrings = self._get_circle()

            result += [ c for c in circle_linestrings ]

        return result

    def _get_circle(self):
        x, y = self.extent.random_point()
        r = self.extent.bounds.exterior.distance(sh.geometry.Point(x, y))

        print(r)

        circle_poly = sh.geometry.Point(x, y).buffer(r)
        print(circle_poly.boundary)

        result = []

        for ls in self.extent.clip_linestring(circle_poly.boundary):
            print(ls)
            result.append(ls)

        return result

class NoOpSymmetrizer:

    def __init__(self, extent):
        self.extent = extent

    def symmetrize(self, lines):
        return lines

class FourPaneSymmetrizer:

    def __init__(self, extent):
        self.extent = extent

    def symmetrize(self, lines):
        print(f"Scaling everything/mirroring")
        lines_next = []
        for line in lines:
            center = extent.center

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

        return lines_next

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", metavar="OUTPUT")
    args = parser.parse_args()

    SVG_WIDTH = 400
    SVG_HEIGHT = 400
    surface = cairo.SVGSurface(args.output, SVG_WIDTH, SVG_HEIGHT)
    c = cairo.Context(surface)
    c.set_line_width(2)

    extent = GeneratorExtent(0, 0, SVG_WIDTH, SVG_HEIGHT)

    shapes = CirclesShapeGenerator(extent).get_linestrings()

    for s in shapes:
        if not (s.type == "LineString" or s.type == "Polygon"):
            raise RuntimeError(f"All shape types must be linestring, but encountered {s}")

    lines = []
    for shape in shapes:
        if shape.type == "LineString":
            lines.append(shape)
        else:
            lines.append(shape.boundary)



    print(f"Splitting {len(lines)} linestrings into crossing lines")
    lines = split_intersections(lines)

    #print(f"Removing horizontal or vertical lines with more than two intersection points.")
    #simplify_intersections(lines)

    lines = NoOpSymmetrizer(extent).symmetrize(lines)

    # Create a graph representation of the lines
    print(f"Creating graph representation")
    graph = get_graph_representation(lines)
    print(graph)

    # Get minimum spanning tree
    print("Creating spanning tree")
    graph = nx.maximum_spanning_tree(graph)

    print("Pruning dingleberries ;)")
    graph = prune_dingleberries(graph)

    # Create a multi-line string and buffer it to generate an outline
    lines = [ graph.edges[e]["line"] for e in graph.edges ]

    mls = sh.geometry.MultiLineString(lines)

    outline = mls.buffer(1, \
            cap_style=sh.geometry.CAP_STYLE.square, \
            join_style=sh.geometry.JOIN_STYLE.mitre)

    polys = []
    print(outline.type)
    if outline.type == "Polygon":
        polys = [ outline ]
    elif outline.type == "MultiPolygon":
        polys = outline.geoms
    else:
        raise RuntimeError()

    lines = []
    for poly in polys:
        print(f"Drawing polygon {poly.boundary.type}")
        lines = poly.boundary.geoms if poly.boundary.type == "MultiLineString" else [ poly.boundary ]

        c.new_path()
        for line in lines:
            coords = [ coord for coord in line.coords ]

            c.set_source_rgba(0,0,0,1)
            c.set_line_cap(cairo.LineCap.ROUND)

            c.move_to(coords[0][0], coords[0][1])
            for coord in coords:
                c.line_to(coord[0], coord[1])

            #c.set_source_rgba(random.uniform(0,0.5),random.uniform(0,0.5),random.uniform(0,0.5),1)
            #c.stroke()

        c.close_path()
        c.set_source_rgba(0,0,0,1)
        c.stroke_preserve()
        c.set_source_rgba(1,0,0,1)
        c.fill()



    print("Saving")

    surface.finish()

    print("Done")

