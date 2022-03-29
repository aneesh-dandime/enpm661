import string
import numpy as np
import argparse
import cv2

from queue import PriorityQueue
from Obstacle import Circle, ClosedFigure
from Point import Point
from Arena import Arena

def correct_theta(theta: float) -> float:
    '''
    Make sure angle is between 0 and pi.
    '''
    while(theta > np.pi):
        theta = theta - 2.0 * np.pi
    while(theta < -np.pi):
        theta = theta + 2.0 * np.pi

    return theta

def degrees(theta: float) -> float:
    '''
    Convert angle to degrees from radians.
    '''
    return (theta * 180 / np.pi)

def radians(theta: float) -> float:
    '''
    Convert angle to radians from degrees.
    '''
    return (theta / 180 * np.pi)

class Graphv2:
    class Node():
        def __init__(self, state: tuple, cost: float, index: int, parent_index: int) -> None:
            self.state = state
            self.cost = cost
            self.index = index
            self.parent_index = parent_index


    def __init__(self, start_state: tuple, goal_state: tuple, occupancy_grid:  np.ndarray, 
                 clearance: int, threshold: float, step_size: float, steer_limit: float, 
                 steer_step: float, make_video: bool) -> None:
        
        self.start_state = start_state
        self.goal_state = goal_state
        self.grid = occupancy_grid
        self.clearance = clearance
        self.threshold = threshold
        self.step_size = step_size
        self.steer_limit = radians(steer_limit)
        self.steer_step = radians(steer_step)
        self.dup_thresh = 0.5

        if self.traversable(start_state, clearance):
            print(f'The start state is not valid!')
            exit()
        if(self.traversable(goal_state, clearance)):
            print(f'The goal state is not valid!')
            exit()

        self.cindex = 0
        self.start_node = self.Node(self.start_state, 0, self.cindex, None)
        self.goal_node = self.Node(self.goal_state, 0, -1, None)

        # Generate steering angles
        self.steering_angles = np.linspace(-self.steer_limit, self.steer_limit, int((self.steer_limit * 2 // self.steer_step) + 1))

        # Open and close lists
        self.olist = dict()
        self.clist = dict()

        # Map to mark visited nodes.
        self.vmap_size = [float(self.grid.shape[0])/self.dup_thresh, float(self.grid.shape[1])/self.dup_thresh, 360.0/degrees(self.steer_step)]
        self.v_map = np.array(np.ones(list(map(int, self.vmap_size))) * np.inf)

        # Tracking variables for plotting path.
        self.final_node = None
        self.path = None
        self.make_video = make_video
        self.iters = 0
        self.num_nodes = 0
        self.total_cost = 0.0
        self.grid_for_video = None

        if(self.make_video):
            self.video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('F','M','P','4'), 24, (self.grid.shape[0], self.grid.shape[1]))

    def traversable(self, state: np.ndarray, clearance: int) -> bool:
        '''
        Check if a node is traversable, i.e., not out of bounds or in an obstacle.
        '''
        points_x, points_y = np.ogrid[max(0, int(state[0]) - clearance): min(self.grid.shape[0], int(state[0]) + clearance), max(0, int(state[1]) - clearance): min(self.grid.shape[1], int(state[1]) + clearance)]
        # Checking if the point is in bounds of the arena
        if (state[0] < 10) or (state[0] > self.grid.shape[0] - 10) or (state[1] < 10) or (state[1] > self.grid.shape[1] - 10):
            return True
        # Checking if the point is inside or outside the obstacle
        elif(not self.grid[int(state[0]),int(state[1])]):
            return True
        elif(len(np.where(self.grid[points_x, points_y] == 0)[0])):
            return True
        else:
            return False

    def check_visited(self, node: Node) -> bool:
        '''
        Check to see if the state is already visited.
        '''
        x = int((round(node.state[0] * 2) / 2) / self.dup_thresh)
        y = int((round(node.state[1] * 2) / 2) / self.dup_thresh)
        theta = round(degrees(node.state[2]))
        if(theta < 0.0):
            theta += 360
        theta = int(theta / 30)
        if(self.v_map[x][y][theta] != np.inf):
            return True
        else:
            return False

    def mark_visited(self, node: Node) -> None:
        '''
        Mark the state as visited.
        '''
        x = int((round(node.state[0] * 2) / 2) / self.dup_thresh)
        y = int((round(node.state[1] * 2) / 2) / self.dup_thresh)
        theta = round(degrees(node.state[2]))
        if(theta < 0.0):
            theta += 360
        theta = int(theta / 30)
        self.v_map[x][y][theta] = node.cost

    def distance(self, state_1: tuple, state_2: tuple) -> float:
        '''
        Euclidean distance between two states.
        '''
        return (np.sqrt(sum((np.asarray(state_1) - np.asarray(state_2)) ** 2)))

    def next_state(self, state: tuple, steering_angle: int) -> tuple:
        '''
        Get the next state by steering.
        '''
        x = state[0] + (self.step_size * np.cos(steering_angle))
        y = state[1] + (self.step_size * np.sin(steering_angle))
        theta = correct_theta(state[2] + steering_angle)
        return (x, y, theta)

    def find_path(self) -> bool:
        '''
        Find the path between self.start_state and self.goal_state and set all variables accordingly.
        Returns true if path found, false otherwise.
        '''
        pq = PriorityQueue()
        pq.put((self.start_node.cost, self.start_node.state))
        self.olist[self.start_node.state] = (self.start_node.index, self.start_node)
        if(self.make_video):
            occupancy_grid = np.uint8(np.copy(self.grid))
            occupancy_grid = cv2.cvtColor(np.flip(np.uint8(occupancy_grid).transpose(), axis=0), cv2.COLOR_GRAY2BGR)
            cv2.circle(occupancy_grid, (self.start_state[0], self.grid.shape[1] - self.start_state[1]), 2, (0, 255, 0), 2)
            cv2.circle(occupancy_grid, (self.goal_state[0], self.grid.shape[1] - self.goal_state[1]), 2, (0, 0, 255), 2)
            self.video.write(np.uint8(occupancy_grid))

        while(not pq.empty()):
            self.iters += 1
            current_node = self.olist[pq.get()[1]][1]
            self.mark_visited(current_node)
            self.clist[current_node.state] = (current_node.index, current_node)
            del self.olist[current_node.state]

            if(self.make_video and self.iters % 500 == 0):
                print(f'Current iteration: {self.iters}')
                try:
                    closed_list_ = dict(self.clist.values())
                    parent_node = closed_list_[current_node.parent_index]
                    start = (int(parent_node.state[0]), (self.grid.shape[1] - 1) - int(parent_node.state[1]))
                    end = (int(current_node.state[0]), (self.grid.shape[1] - 1) - int(current_node.state[1]))
                    cv2.line(occupancy_grid, start, end, (255,0,0), 1)
                    if(self.iters % 500 == 0):
                        self.video.write(np.uint8(occupancy_grid))
                except Exception as e:
                    print(e)

            if(self.distance(current_node.state[:2], self.goal_state[:2]) <= self.threshold):
                print(f'Reached destination! Iterations: {self.iters}')
                self.final_node = current_node
                if(self.make_video):
                    self.grid_for_video = occupancy_grid
                self.total_cost = current_node.cost
                return True

            for steering_angle in self.steering_angles:
                new_state = self.next_state(current_node.state, steering_angle)
                new_index = self.cindex + 1
                self.cindex = new_index
                new_cost = current_node.cost + self.step_size
                if(not self.traversable(new_state, self.clearance)):
                    new_node = self.Node(new_state, new_cost, new_index, current_node.index)
                    if(self.check_visited(new_node)):
                        self.cindex -= 1
                        continue
                    if(new_state in self.clist):
                        self.cindex -= 1
                        continue
                    if(new_state not in self.olist):
                        self.olist[new_state] = (new_node.index, new_node)
                        pq.put((new_node.cost + self.distance(new_state[:2], self.goal_state[:2]), new_node.state))
                    else:
                        if(self.olist[new_state][1].cost > new_node.cost):
                            self.olist[new_state] = (new_node.index, new_node)
                        else:
                            self.cindex -= 1
                    self.num_nodes += 1
                else:
                    self.cindex -= 1
                    pass
        
        print(f'Goal node not reachable with given conditions!')
        return False

    def backtrack_path(self) -> np.ndarray:
        '''
        Backtrack and find the actual path.
        '''
        current_node = self.final_node
        self.path = list()
        traversed_nodes = dict(self.clist.values())
        while(current_node.index != 0):
            self.path.append(current_node.state)
            current_node = traversed_nodes[current_node.parent_index]
        self.path.append(self.start_node.state)
        self.path.reverse()
        print(f'The length of the path is: {len(self.path)}')
        self.path = np.array(self.path).astype(int)
        self.grid_for_video = cv2.circle(self.grid_for_video, (self.start_state[0], self.grid.shape[1] - self.start_state[1]), 2, (0, 255, 0), 2)
        self.grid_for_video = cv2.circle(self.grid_for_video, (self.goal_state[0], self.grid.shape[1] - self.goal_state[1]), 2, (0, 0, 255), 2)
        if(self.make_video):
            for step in range(len(self.path)-1):
                self.grid_for_video = cv2.line(self.grid_for_video, (self.path[step,0], self.grid.shape[1] - self.path[step,1]), (self.path[step+1,0], self.grid.shape[1] - self.path[step+1,1]), (0,0,255), 2)
                self.video.write(np.uint8(self.grid_for_video))
        print(f'Path Length: {self.total_cost}')
        return self.path
    
def string_to_list(st: str) -> list:
    return list(map(int, st.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_state', type=str, default="[10, 10, 0]")
    parser.add_argument('--goal_state', type=str, default="[50, 50, 0]")
    parser.add_argument('--threshold', type=float, default=1.5)
    parser.add_argument('--step_size', type=float, default=1.0)
    parser.add_argument('--steer_limit', type=float, default=60.0)
    parser.add_argument('--steer_step', type=float, default=30.0)
    parser.add_argument('--make_video', type=bool, default=False)
    args = parser.parse_args()
    start_state = string_to_list(args.start_state)
    start_state[2] = radians(start_state[2])
    start_state = tuple(start_state)
    goal_state = string_to_list(args.goal_state)
    goal_state[2] = radians(goal_state[2])
    goal_state = tuple(goal_state)
    threshold = args.threshold
    step_size = args.step_size
    steer_limit = args.steer_limit
    steer_step = args.steer_step
    make_video = args.make_video

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

    graph = Graphv2(start_state, goal_state, arena.grid, 15, threshold, step_size, steer_limit, steer_step, make_video)
    if(graph.find_path()):
        print(f'Path found...backtracking...')
        graph.backtrack_path()