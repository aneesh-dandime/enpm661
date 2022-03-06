"""
Finding shortest path through a given map,
using Dijktra's algorithm.
"""
import argparse
from Point import Point
from Obstacle import Circle, ClosedFigure
from Arena import Arena
from random import seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_x', type=float, help='The x coordinate of the source.')
    parser.add_argument('--src_y', type=float, help='The y coordinate of the source.')
    parser.add_argument('--dst_x', type=float, help='The x coordinate of the destination.')
    parser.add_argument('--dst_y', type=float, help='The y coordinate of the destination.')
    parser.add_argument('--name', type=str, help='Name of the video file.')
    args = parser.parse_args()
    src = Point(args.src_x, args.src_y)
    dst = Point(args.dst_x, args.dst_y)
    name = args.name
    
    seed(1)
    circle = Circle(Point(300.0, 185.0), 40.0, 5.0)
    hexagon = ClosedFigure([
        Point(200 - 35, 100 + 20.207),
        Point(200, 100 + 40.415),
        Point(200 + 35, 100 + 20.207),
        Point(200 + 35, 100 - 20.207),
        Point(200, 100 - 40.415),
        Point(200 - 35, 100 - 20.207)
    ], 5.)
    closed_figure = ClosedFigure([
        Point(105, 100),
        Point(36, 185),
        Point(115, 210),
        Point(80, 180)
    ], 5.)
    obstacles = [circle, hexagon, closed_figure]
    arena = Arena(400.0, 250.0, obstacles)
    arena.shortest_path(src, dst, name)

