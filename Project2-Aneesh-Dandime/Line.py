import math
from Point import Point
from typing import TypeVar


TLine = TypeVar('TLine', bound='Line')


class Line:
    def __init__(self, a: Point, b: Point) -> None:
        self.a = a
        self.b = b
    
    def distance(self, p: Point) -> float:
        vector1 = (self.b.x - self.a.x, self.b.y - self.a.y)
        vector2 = (p.x - self.a.x, p.y - self.a.y)
        vector3 = (self.b.x - p.x, self.b.y - p.y)
        r = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        r = r / (vector1[0] ** 2 + vector1[1] ** 2)

        if r < 0:
            return math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        elif r > 1:
            return math.sqrt(vector3[0] ** 2 + vector3[1] ** 2)
        else:
            return math.sqrt(abs((vector2[0] ** 2 + vector2[1] ** 2) - (r * math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)) ** 2))
    
    @staticmethod
    def intersect(l1: TLine, l2: TLine) -> bool:
        p1 = l1.a
        q1 = l1.b
        p2 = l2.a
        q2 = l2.b
        o1 = Point.orientation(p1, q1, p2)
        o2 = Point.orientation(p1, q1, q2)
        o3 = Point.orientation(p2, q2, p1)
        o4 = Point.orientation(p2, q2, q1)
    
        if ((o1 != o2) and (o3 != o4)):
            return True
        if ((o1 == 0) and Point.on_segment(p1, p2, q1)):
            return True
        if ((o2 == 0) and Point.on_segment(p1, q2, q1)):
            return True
        if ((o3 == 0) and Point.on_segment(p2, p1, q2)):
            return True
        if ((o4 == 0) and Point.on_segment(p2, q1, q2)):
            return True
    
        return False
    
    def __str__(self) -> str:
        return f'Line from {self.a} to {self.b}.'
