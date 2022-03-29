from typing import List, Tuple
from Obstacle import Obstacle
from Point import Point
import numpy as np


class Arena:
    def __init__(self, width: float, height: float, obstacles: List[Obstacle]) -> None:
        self.width = int(width) + 1
        self.height = int(height) + 1
        self.obstacles = []
        for obstacle in obstacles:
            self.obstacles.append(obstacle)
        self.grid = self.__build_grid()

    def __build_grid(self) -> np.ndarray:
        grid = np.zeros((self.width, self.height))
        traversable_nodes = 0
        for w in range(self.width):
            for h in range(self.height):
                outside = all(map(lambda obs: obs.is_outside(Point(w, h)), self.obstacles))
                clear = all(map(lambda obs: obs.is_clear(Point(w, h)), self.obstacles))
                if outside and not clear:
                    grid[w, h] = 0
                elif outside and clear:
                    grid[w, h] = 255
                    traversable_nodes += 1
                else:
                    grid[w, h] = 0
        print(f'The grid has {traversable_nodes} traversable points')

        return grid
