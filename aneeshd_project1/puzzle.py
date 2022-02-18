import numpy as np
import re

from typing import Dict, List, Set, Tuple, TypeVar
from queue import Queue


TNode = TypeVar('TNode', bound='Node')
class Node:
    def __init__(self, state: np.ndarray, index: int, pindex: int) -> None:
        self.state = state
        self.index = index
        self.pindex = pindex
    
    def set_index(self, index: int) -> None:
        self.index = index
    
    def empty_location(self) -> Tuple[int, int]:
        loc = np.where(self.state == 0)
        return loc[0][0], loc[1][0]
    
    def swap(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> None:
        self.state[loc1], self.state[loc2] = self.state[loc2], self.state[loc1]
    
    def children(self) -> List[TNode]:
        empty_row, empty_col = self.empty_location()
        clist = []

        if empty_row - 1 >= 0:
            cnode = Node(np.copy(self.state), -1, self.index)
            cnode.swap((empty_row, empty_col), (empty_row - 1, empty_col))
            clist.append(cnode)
        
        if empty_row + 1 < self.state.shape[0]:
            cnode = Node(np.copy(self.state), -1, self.index)
            cnode.swap((empty_row, empty_col), (empty_row + 1, empty_col))
            clist.append(cnode)
        
        if empty_col - 1 >= 0:
            cnode = Node(np.copy(self.state), -1, self.index)
            cnode.swap((empty_row, empty_col), (empty_row, empty_col - 1))
            clist.append(cnode)
        
        if empty_col + 1 < self.state.shape[1]:
            cnode = Node(np.copy(self.state), -1, self.index)
            cnode.swap((empty_row, empty_col), (empty_row, empty_col + 1))
            clist.append(cnode)
        
        return clist
    
    def solvable(self) -> bool:
        lstate = self.state[np.where(self.state != 0)].flatten()
        inv = 0
        for i in range(lstate.shape[0]):
            for j in range(i + 1, lstate.shape[0]):
                if lstate[i] > lstate[j]:
                    inv += 1
        return True if inv % 2 == 0 else False
    
    def is_goal(self, goal_state) -> bool:
        return np.equal(self.state, goal_state).all()
    
    def __hash__(self) -> int:
        return hash(tuple(self.state.flatten()))
    
    def __str__(self) -> str:
        return str(self.state)


class EightPuzzle:    
    def __init__(self, init_state: np.ndarray) -> None:
        self.init_state = init_state
        self.nodes = [Node(self.init_state, 0, -1)]
        self.solved = False
        self.goal_index = -1
        self.goal_state = np.array([[1,2,3], [4,5,6], [7,8,0]])
    
    def solve(self) -> bool:
        if self.solved:
            return True
        
        q: Queue[TNode] = Queue()
        q.put(self.nodes[0])
        visited: Set[int] = set()

        while not q.empty():
            top_node = q.get()
            if hash(top_node) in visited:
                continue
            visited.add(hash(top_node))
            
            if not top_node.solvable():
                continue
            if top_node.is_goal(self.goal_state):
                self.solved = True
                self.goal_index = top_node.index
                print(self.nodes[self.goal_index])
                return True

            cnodes = top_node.children()
            for cnode in cnodes:
                if not hash(cnode) in visited:
                    cnode.set_index(len(self.nodes))
                    self.nodes.append(cnode)
                    q.put(cnode)
        
        return False
    
    def get_path(self) -> List[TNode]:
        if not self.solved:
            return []
        
        path = []
        node = self.nodes[self.goal_index]
        path.append(node)
        
        while node.pindex != -1:
            node = self.nodes[node.pindex]
            path.append(node)
        
        path.reverse()
        return path


def print_path(path: List[TNode]) -> None:
    if not path or len(path) == 0:
        print('Path is empty!')
    print('\n==== The path ====\n')
    for i, node in enumerate(path):
        print(node)

        if i != len(path) - 1:
            print('==== | ====')
            print('==== | ====')
            print('==== V ====')
    print('\n==== END ====\n')


def generate_files(puzzle: EightPuzzle, path: List[TNode]) -> None:
    nodes_file = 'Nodes.txt'
    nodes_info_file = 'NodesInfo.txt'
    node_path_file = 'nodePath.txt'

    nodes_str = ''
    for node in puzzle.nodes:
        rows, cols = node.state.shape
        for col in range(cols):
            for row in range(rows):
                nodes_str = nodes_str + f'{node.state[row, col]} '
        nodes_str = nodes_str + '\n'
    
    nodes_path = ''
    for node in path:
        rows, cols = node.state.shape
        for col in range(cols):
            for row in range(rows):
                nodes_path = nodes_path + f'{node.state[row, col]} '
        nodes_path = nodes_path + '\n'

    with open(nodes_file, 'w') as f:
        f.write(nodes_str)
    
    with open(nodes_info_file, 'w') as f:
        f.write('Node_index   Parent_Node_index   Cost\n')
        for node in puzzle.nodes:
            node_info = f'{node.index + 1} {node.pindex + 1} 0\n'
            f.write(node_info)
    
    with open(node_path_file, 'w') as f:
        f.write(nodes_path)


def get_state_np(state:  str) -> np.ndarray:
    pattern = r'^\[(\d),(\d),(\d)\],\[(\d),(\d),(\d)\],\[(\d),(\d),(\d)\]$'
    m = re.search(pattern, state)
    
    if not m:
        raise ValueError('The provided input is not in correct format!')
    
    nums = [int(group) for group in m.groups()]
    np_state = np.array(nums).reshape((3, 3)).transpose()
    return np_state


def main():
    state = input('Enter state: ').strip()
    np_state = get_state_np(state)
    eight_puzzle = EightPuzzle(np_state)

    if eight_puzzle.solve():
        print('Solved!')
    else:
        print('Cannot solve!')
    
    path = eight_puzzle.get_path()
    generate_files(eight_puzzle, path)


if __name__ == '__main__':
    main()

