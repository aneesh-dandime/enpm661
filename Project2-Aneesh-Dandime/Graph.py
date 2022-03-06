from typing import List, Tuple
from queue import PriorityQueue
from Point import Point
import numpy as np
import math as m
import time
import cv2
import matplotlib.pyplot as plt


class Graph:
    class Node:
        def __init__(self, index: int, x: int, y: int) -> None:
            self.index = index
            self.children = []
            self.parent_index = -1
            self.cost = 100000000
            self.x = x
            self.y = y

        def add_child(self, index: int, cost: float) -> None:
            self.children.append((index, cost))

    def __init__(self, grid: List[List[Tuple[int, int]]]) -> None:
        if len(grid) <= 0:
            raise ValueError('The grid is empty!')

        self.grid = grid
        self.adj_list = None
        self.__build_graph()

    def __build_graph(self) -> None:
        self.adj_list = dict()
        width = len(self.grid)
        height = len(self.grid[0])
        traversable_nodes = 0

        for w in range(width):
            for h in range(height):
                index = self.grid[w][h][0]
                traversable = self.grid[w][h][1]
                
                if traversable == 1:
                    traversable_nodes += 1
                    self.adj_list[index] = self.Node(index, w, h)
                    node = self.adj_list[index]

                    if w - 1 >= 0 and h - 1 >= 0 and self.grid[w-1][h-1][1] == 1:
                        node.add_child(self.grid[w-1][h-1][0], m.sqrt(2))
                    if h - 1 >= 0 and self.grid[w][h-1][1] == 1:
                        node.add_child(self.grid[w][h-1][0], 1.)
                    if w + 1 < width and h - 1 >= 0 and self.grid[w+1][h-1][1] == 1:
                        node.add_child(self.grid[w+1][h-1][0], m.sqrt(2))
                    if w - 1 >= 0 and self.grid[w-1][h][1] == 1:
                        node.add_child(self.grid[w-1][h][0], 1.)
                    if w + 1 < width and self.grid[w+1][h][1] == 1:
                        node.add_child(self.grid[w+1][h][0], 1.)
                    if w - 1 >= 0 and h + 1 < height and self.grid[w-1][h+1][1] == 1:
                        node.add_child(self.grid[w-1][h+1][0], m.sqrt(2))
                    if h + 1 < height and self.grid[w][h+1][1] == 1:
                        node.add_child(self.grid[w][h+1][0], 1.)
                    if w + 1 < width and h + 1 < height and self.grid[w+1][h+1][1] == 1:
                        node.add_child(self.grid[w+1][h+1][0], m.sqrt(2))
        
        print(f'The graph has {len(self.adj_list)} nodes.')

    def shortest_path(self, p: Point, q: Point, name: str) -> None:
        src_index = self.grid[int(p.x)][int(p.y)][0]
        dst_index = self.grid[int(q.x)][int(q.y)][0]
        explore_order = []

        visited = set()
        pq = PriorityQueue()
        pq.put((0, src_index))

        start_time = time.time()

        width = len(self.grid)
        height = len(self.grid[0])

        while not pq.empty():
            top_ = pq.get()
            top_distance = top_[0]
            top_index = top_[1]
            top_node = self.adj_list[top_index]
            
            if top_index in visited:
                continue
            else:
                visited.add(top_index)
                explore_order.append(top_index)

            if top_index == dst_index:
                time_taken = time.time() - start_time
                print(f'The length of the shortest path from {p} to {q} is {top_distance}.')
                print(f'Took {time_taken}s to find shortest path.')
                path = self.get_path(dst_index)
                self.animate(explore_order, src_index, dst_index, path, name)
                return

            for child in top_node.children:
                if child[0] not in visited:
                    child_node = self.adj_list[child[0]]
                    if child_node.cost > child[1] + top_distance:
                        child_node.cost = child[1] + top_distance
                        child_node.parent_index = top_index
                        pq.put((child[1] + top_distance, child[0]))

        print(f'There is no path from {p} to {q}.')
    
    def animate(self, explore_order: List[int], src: int, dst: int, path: List[int], name: str) -> None:
        width = len(self.grid)
        height = len(self.grid[0])
        base_frame = np.zeros((width, height))
        frames = []
        src_x = self.adj_list[src].x
        src_y = self.adj_list[src].y
        dst_x = self.adj_list[dst].x
        dst_y = self.adj_list[dst].y
        
        for w in range(width):
            for h in range(height):
                if self.grid[w][h][1] == 0:
                    base_frame[w, h] = 0
                elif self.grid[w][h][1] == 1:
                    base_frame[w, h] = 255
                else:
                    base_frame[w, h] = 200
        base_frame[src_x][src_y] = 125
        base_frame[dst_x][dst_y] = 125

        frames.append(np.copy(base_frame))

        iter = 0
        step = 20
        for index in explore_order:
            x = self.adj_list[index].x
            y = self.adj_list[index].y
            base_frame[x][y] = 125

            if iter % step == 0:
                frames.append(np.copy(base_frame))
            
            iter += 1
        
        final_frame = np.copy(base_frame)
        for i in range(len(path) - 1):
            index1 = path[i]
            index2 = path[i + 1]
            x1 = self.adj_list[index1].x
            y1 = self.adj_list[index1].y
            x2 = self.adj_list[index2].x
            y2 = self.adj_list[index2].y
            final_frame = cv2.line(final_frame, (y1, x1), (y2, x2), 0, 2)
        for _ in range(100):
            frames.append(final_frame)
        
        for frame in frames:
            cv2.imshow('Arena', np.uint8(np.flip(np.transpose(frame), axis=0)))
            cv2.waitKey(1)
        
        video = cv2.VideoWriter(f'{name}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 100, (width, height), 0)
        for frame in frames:
            video.write(np.uint8(np.flip(np.transpose(frame), axis=0)))
        video.release()

    
    def get_path(self, dst_index: int) -> List[int]:
        current = dst_index
        path = [dst_index]
        while self.adj_list[current].parent_index != -1:
            current = self.adj_list[current].parent_index
            path.append(current)
        path.reverse()
        return path

