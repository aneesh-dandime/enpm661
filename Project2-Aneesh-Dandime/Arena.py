from typing import List
from Obstacle import Obstacle
from Graph import Graph
from Point import Point


class Arena:
    def __init__(self, width: float, height: float, obstacles: List[Obstacle]) -> None:
        self.width = int(width) + 1
        self.height = int(height) + 1
        self.obstacles = []
        for obstacle in obstacles:
            self.obstacles.append(obstacle)
        self.grid = None
        self.__build_grid()
        self.graph = Graph(self.grid)

    def __build_grid(self) -> None:
        self.grid = []
        index = 0
        traversable_nodes = 0
        for w in range(self.width):
            self.grid.append([])
            for h in range(self.height):
                outside = all(map(lambda obs: obs.is_outside(Point(w, h)), self.obstacles))
                clear = all(map(lambda obs: obs.is_clear(Point(w, h)), self.obstacles))
                if outside and not clear:
                    self.grid[w].append((index, 2))
                elif outside and clear:
                    self.grid[w].append([index, 1])
                    traversable_nodes += 1
                else:
                    self.grid[w].append((index, 0))
                index += 1
        print(f'The grid has {traversable_nodes} traversable points')

    def add_obstacle(self, obs: Obstacle) -> None:
        self.obstacles.append(obs)
        self.__build_graph()
    
    def shortest_path(self, p: Point, q: Point, name: str) -> None:
        width = len(self.grid)
        height = len(self.grid[0])

        if p.x < 0 or p.x >= width or p.y < 0 or p.y >= height:
            raise ValueError(f'The point {p} is not in the arena!')
        if q.x < 0 or q.x >= width or q.y < 0 or q.y >= height:
            raise ValueError(f'The point {q} is not in the arena!')
        if self.grid[int(p.x)][int(p.y)][1] == 0:
            raise ValueError(f'The point {p} is inside an obstacle!')
        if self.grid[int(q.x)][int(q.y)][1] == 0:
            raise ValueError(f'The point {q} is inside an obstacle!')

        self.graph.shortest_path(p, q, name)

