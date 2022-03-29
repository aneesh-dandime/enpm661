from abc import ABC, abstractmethod
from typing import List
from Point import Point
from Line import Line
from random import randint


class Obstacle(ABC):
    @abstractmethod
    def is_outside(self, p: Point) -> bool:
        pass

    @abstractmethod
    def is_clear(self, p: Point) -> bool:
        pass


class Circle(Obstacle):
    def __init__(self, c: Point, r: float, d: float) -> None:
        self.c = c
        self.r = r
        self.d = d

    def is_outside(self, p: Point) -> bool:
        return self.c.distance(p) > self.r

    def is_clear(self, p: Point) -> bool:
        return self.c.distance(p) > (self.r + self.d)


class ClosedFigure(Obstacle):
    def __init__(self, points: List[Point], d: float) -> None:
        self.points = points
        self.lines = [Line(self.points[i], self.points[i + 1]) for i in range(len(points) - 1)]
        self.lines.append(Line(points[-1], points[0]))
        self.d = d
    
    def is_outside(self, p: Point) -> bool:
        ref_line = Line(p, Point(randint(401, 10000), randint(251, 10000)))
        intersections = 0
        for line in self.lines:
            if Line.intersect(line, ref_line):
                intersections += 1
        return intersections % 2 == 0
    
    def is_clear(self, p: Point) -> bool:
        return self.is_outside(p) and (min(map(lambda l: l.distance(p), self.lines)) > self.d)

