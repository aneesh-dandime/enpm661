import math as m
from typing import TypeVar


TPoint = TypeVar('TPoint', bound='Point')


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def distance(self, p: TPoint) -> float:
        return m.sqrt((self.y - p.y) ** 2 + (self.x - p.x) ** 2)

    def __str__(self) -> str:
        return f'({self.x}, {self.y})'
    
    @staticmethod
    def on_segment(p: TPoint, q: TPoint, r: TPoint) -> bool:
        if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
            return True
        return False
    
    @staticmethod
    def orientation(p: TPoint, q: TPoint, r: TPoint) -> int:
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
        if val > 0:
            return 1
        elif val < 0:
            return 2
        else:
            return 0