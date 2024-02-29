import sys
import datetime
from collections import deque
from queue import Queue, PriorityQueue


#BFS

import hashlib

class Node:
    def __init__(self, state, parent, move, cost, depth):
        self.state = state
        self.parent = parent
        self.move = move
        self.cost = cost
        self.depth = depth

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        state_bytes = str(self.state).encode('utf-8')
        return int(hashlib.sha256(state_bytes).hexdigest(), 16)

        
def node_bfs(node):
    subchild = []
    movepos = ["Up", "Down", "Left", "Right"]
    startindex = node.state.index(0)
    row, col = divmod(startindex, 3)

    for move in movepos:
        if move == "Up":
            move_row, move_col = row - 1, col
        elif move == "Down":
            move_row, move_col = row + 1, col
        elif move == "Right":
            move_row, move_col = row, col + 1
        else:
            move == "Left"
            move_row, move_col = row, col - 1

        if not (0 <= move_row <= 2 and 0 <= move_col <= 2):
            continue

        new_indexmove = move_row * 3 + move_col
        new_st = list(node.state)
        new_st[startindex], new_st[new_indexmove] = new_st[new_indexmove], new_st[startindex]
        cost = node.state[new_indexmove]
        new_node = Node(tuple(new_st), node, move, cost, node.depth + 1)
        subchild.append(new_node)

    return subchild

def get_successors(node):
    successors = []
    i, j = get_blank_pos(node.state)
    if i > 0:
        state = swap(node.state, i, j, i-1, j)
        successors.append(Node(state, "Move {} Down \n".format(state[i][j]), node.g+1, node.d+1, node))
    if i < 2:
        state = swap(node.state, i, j, i+1, j)
        successors.append(Node(state, "Move {} Up".format(state[i][j]), node.g+1, node.d+1, node))
    if j > 0:
        state = swap(node.state, i, j, i, j-1)
        successors.append(Node(state, "Move {} Right".format(state[i][j]), node.g+1, node.d+1, node))
    if j < 2:
        state = swap(node.state, i, j, i, j+1)
        successors.append(Node(state, "Move {} Left".format(state[i][j]), node.g+1, node.d+1, node))
    return successors


def bfs(start, goal):
    visited = set()
    q = Queue()
    root = Node(start, None, None, 0,0)
    q.put(root)
    visited.add(root)
    nodespopped = 0
    maxfringesize = 0
    solution_found = False
    nodesexpanded = 0
   
    nodesgenerated = 0
    closednode = set()

    if dump == 'true':
        file = open(date_string,'a')
    
    while not q.empty():
        maxfringesize = max(maxfringesize, q.qsize())
        node = q.get()
        nodespopped += 1
        
        if node.state == goal:
            solution_found = True
            depth = 0
            steps = []
            cost = 0
            
            while node.parent is not None:
                steps.append((node.move, node.cost))
                cost += node.cost
                depth += node.depth
                node = node.parent
                
            steps.reverse()
            
            return nodespopped, nodesexpanded, nodesgenerated, maxfringesize, depth, cost, steps
        if dump == 'true':
            file.write(f"Generating successors to < state = {node.state}, action = {node.move} g(n) = {node.cost}, d = {node.depth}, f(n) = {node.cost + node.depth} >:")
        subchild = node_bfs(node)
        nodesexpanded += 1
        closednode.add(node.state)

        fringestate = "[\n"
        for child in subchild:
            nodesgenerated += 1
            
            if child not in closednode:
                q.put(child)
                visited.add(child)
                if dump == 'true':
                    fringestate += f"< state = {child.state}, action = {child.move}, g(n) = {child.cost}, d = {child.depth}, f(n) = {child.cost + child.depth}, Parent = Pointer to {child.parent.state} >\n"
        
        if dump == 'true':
            fringestate +="]"
            file.write(f"{len(subchild)} successors generated\nClosed: {list(closednode)}\tFringe: {fringestate}\n")


#Depth first Search

class NodeD:
    def __init__(self, state, parent=None, cost=0, action=None, depth=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.action = action
        self.depth = depth
        self.subchild = []

    def add_child(self, child_node):
        self.subchild.append(child_node)

    def is_leaf(self):
        return len(self.subchild) == 0

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(str(self.state))


def dfs(start, goal):
    start_node = NodeD(start, None, 0, None)  # create start node
    maxfringesize = 0
    nodesexpanded = 0
    nodesgenerated = 0
    nodespopped = 0
    frontier = [start_node]  # initialize frontier with start node
    explored = set()  # initialize explored set

    # open file for writing if dump mode is on
    if dump == 'true':
        file = open(date_string, 'a')
    
    # search until frontier is empty
    while frontier:
        maxfringesize = max(maxfringesize, len(frontier))  # update max frontier size
        node = frontier.pop()  # get node from frontier
        nodespopped += 1  # update nodes popped count
        steps = []  # initialize steps list

        # write to file if dump mode is on
        if dump == 'true':
            file.write(f"Expanding: {node.state}")

        # check if node is goal node
        if node.state == goal:
            steps = []
            while node.parent is not None:
                steps.append((node.action, node.cost))
                node = node.parent
            steps.reverse()
            return nodespopped, nodesexpanded, nodesgenerated, maxfringesize, len(steps), sum(cost for _, cost in steps), steps
        
        # add node to explored set and update nodes expanded count
        explored.add(tuple(node.state))
        nodesexpanded += 1

        # generate successors for current node
        successors = successordfs(node.state)

        # write to file if dump mode is on
        
        if dump == 'true':
            file.write(f"\nGenerating successors to < state = {node.state}, action = {node.action} g(n) = {node.cost}, d = {len(steps)}, f(n) = {node.cost + len(steps)}, Parent = Pointer to {{{node.parent}}}>:")
            file.write(f"\n{len(successors)} successors generated")
            file.write(f"\nClosed: {list(explored)}")
            file.write(f"\nFringe: [")
        
        # add unexplored successors to frontier and update nodes generated count
        for action, cost, successor_node in successors:
            if tuple(successor_node) not in explored:
                nodesgenerated += 1
                child_node = NodeD(successor_node, node, cost, action)
                frontier.append(child_node)
                if dump == 'true':
                    file.write(f"\n\\t< state = {successor_node}, action = {action} g(n) = {node.cost + cost}, d = {len(steps) + 1}, f(n) = {node.cost + cost + len(steps) + 1}, Parent = Pointer to {{{node.state}}}>")
        if dump == 'true':
            file.write("\n]" )  
    
    return None

def successordfs(state):
    successors = []
    startindex = state.index(0)
    movepos = [        ('Down', startindex - 3),        ('Up', startindex + 3),        ('Right', startindex - 1),        ('Left', startindex + 1),    ]
    for move, new_indexmove in movepos:
        if new_indexmove < 0 or new_indexmove >= len(state):
            continue
        new_st = state[:]
        new_st[startindex], new_st[new_indexmove] = new_st[new_indexmove], new_st[startindex]
        successors.append((move, new_st[startindex], new_st))
    return successors


            
#UCS

def indexucs(state):
    blank_positions = [(i, j) for i, row in enumerate(state) for j, val in enumerate(row) if val == 0]
    if len(blank_positions) != 1:
        raise ValueError(f"Expected 1 blank position, found {len(blank_positions)}")
    return blank_positions[0]

def get_ucspath(state):
    ROWS, COLS = 3, 3
    action = []
    # Find the blank tile location
    blank_row, blank_col = next((r, c) for r in range(ROWS) for c in range(COLS) if state[r][c] == 0)

    # Determine the valid actions based on the blank tile location
    if blank_row > 0:
        action.append('Up')
    if blank_row < ROWS - 1:
        action.append('Down')
    if blank_col > 0:
        action.append('Left')
    if blank_col < COLS - 1:
        action.append('Right')
        
    return action



def getucsst(state, action):
    row, col = indexucs(state)
    new_st = [row[:] for row in state]
    weight = 0
    blank_row, blank_col = indexucs(state)
    ROWS, COLS = 3, 3
    return_action = ''
    if action == 'Up' and blank_row > 0:
        return_action = "Down"
        new_st[blank_row][blank_col], new_st[blank_row-1][blank_col] = new_st[blank_row-1][blank_col], new_st[blank_row][blank_col]

    elif action == 'Down' and blank_row < ROWS - 1:
        return_action = "Up"
        new_st[blank_row][blank_col], new_st[blank_row+1][blank_col] = new_st[blank_row+1][blank_col], new_st[blank_row][blank_col]

    elif action == 'Left' and blank_col > 0:
        return_action = "Right"
        new_st[blank_row][blank_col], new_st[blank_row][blank_col-1] = new_st[blank_row][blank_col-1], new_st[blank_row][blank_col]

    elif action == 'Right' and blank_col < COLS - 1:
        return_action = "Left"
        new_st[blank_row][blank_col], new_st[blank_row][blank_col+1] = new_st[blank_row][blank_col+1], new_st[blank_row][blank_col]

    return new_st, return_action


def ucs(initial_state, goal_state):
    
    nodesexpanded = 0
    nodesgenerated = 0
    maxfringesize = 0
    nodespopped = 0
    depth = 0
    cost = 0
    steps = []
    weight = []
    
    frontier = PriorityQueue()
    frontier.put((0,None ,(initial_state, [],[])))
    
    explored = set()
    if dump == 'true':
        file = open(date_string,'a')
        file.write(f"Generating successors to < state = {initial_state}, action = {{Start}} g(n) = 0, d = 0, f(n) = 0, Parent = Pointer to {{None}} >:\n")
        file.write(f"{len(get_ucspath(initial_state))} successors generated\n")
        file.write(f"Closed: []\n")
    
    while not frontier.empty():
        maxfringesize = max(maxfringesize, frontier.qsize())
        
        present_cost,action_state ,(present_state, present_path, present_weight) = frontier.get()
        nodespopped += 1
        
        if goal_state == present_state:
            nodesexpanded = len(explored)
            depth = len(present_path)
            weight = present_weight
            cost = present_cost
            steps = present_path
            
            break
        
        explored.add(tuple(map(tuple, present_state)))
        
        
        actions = get_ucspath(present_state)
        
        
        for action in actions:
            new_st, action_state = getucsst(present_state, action)
            new_cost = present_cost + new_st[indexucs(present_state)[0]][indexucs(present_state)[1]]
            new_weight = new_st[indexucs(present_state)[0]][indexucs(present_state)[1]]
            if tuple(map(tuple, new_st)) not in explored:
                frontier.put((new_cost,action_state, (new_st, present_path + [action_state], present_weight + [new_weight])))
                nodesgenerated += 1
            if dump == 'true':
                file.write(f"\nGenerating successors to < state = {present_state}, action = {action_state} g(n) = {present_cost}, d = {len(present_path)}, f(n) = {present_cost}, Parent = Pointer to {{{'None' if len(present_path) == 0 else present_path[-1]}}}>:")
                file.write(f"\n{len(actions)} successors generated")
                file.write(f"\nClosed: {explored}")
                file.write("\nFringe: [")
                for item in frontier.queue:
                    file.write(f"\n< state = {item[2][0]}, action = {action} g(n) = {new_cost}, d = {len(item[2])}, f(n) = {new_cost + new_weight}, Parent = Pointer to {new_st}>")
                file.write("\n]\n")
    step = []
    for i in range(len(steps)):
        step.append([steps[i],weight[i]])

    return nodespopped, nodesexpanded, nodesgenerated, maxfringesize, depth, cost, step




#Depth Limited Search'

def dls(start, goal, limit):
    class Node:
        def __init__(self, state, parent, move, cost,weight):
            self.state = state
            self.parent = parent
            self.move = move
            self.cost = cost
            self.weight = weight
    
    def get_movep(state):
        movepos = []
        
        empty_pos = state.index(0)
        if empty_pos % 3 > 0:
            movepos.append(('Left', state[empty_pos-1]))
        if empty_pos % 3 < 2:
            movepos.append(('Right', state[empty_pos+1]))
        if empty_pos // 3 > 0:
            movepos.append(('Up', state[empty_pos-3]))
        if empty_pos // 3 < 2:
            movepos.append(('Down', state[empty_pos+3]))
        return movepos
    
    explored = set()
    root = Node(start, None, None, 0,0)
    frontier = [(root, 0)]
    nodespopped = 0
    maxfringesize = 0
    nodesexpanded = 0
    
    move_ch = {'Left': -1, 'Right': 1, 'Up': -3, 'Down': 3}
    

    if dump == 'true':
        file = open(date_string,'a')

    while frontier:
        if len(frontier) > maxfringesize:
            maxfringesize = len(frontier)
        node, depth = frontier.pop()
        nodespopped += 1
        if dump == 'true':
            file.write(f"Generating successors to < state = {node.state}, action = {node.move} g(n) = {node.cost}, d = {depth}, f(n) = {node.cost + depth}, Parent = Pointer to {None} >:\n")
    
        if node.state == goal:
            steps = []
            cost = node.cost
            
            while node.parent is not None:
                if(node.move == "Down"):
                    node.move = "Up"
                elif(node.move == "Left"):
                    node.move = "Right"
                elif(node.move == "Up"):
                    node.move = "Down"
                
                else:
                    node.move = "Left"
                steps.append((node.move, node.weight))
                node = node.parent
            steps.reverse()
            return nodespopped, nodesexpanded, len(explored), maxfringesize, len(steps), cost, steps
        
        if depth < limit:
            explored.add(tuple(node.state))
            nodesexpanded += 1
            for move, next_state_cost in get_movep(node.state):
                next_state = node.state.copy()
                empty_pos = next_state.index(0)
                next_state[empty_pos] = next_state_cost
                next_state[empty_pos + move_ch[move]] = 0
                weight = next_state_cost
                if tuple(next_state) not in explored:
                    child = Node(next_state, node, move, node.cost + next_state_cost,weight)
                    frontier.append((child, depth+1))
                    if dump == 'true':
                        file.write(f"< state = {child.state}, action = {child.move} g(n) = {child.cost}, d = {depth+1}, f(n) = {child.cost + depth + 1}, Parent = Pointer to {child.parent.state} >\n")
    
    return None

def dlss(start, goal, depth_limit):
    result = None
    for i in range(depth_limit, sys.maxsize):
        result = dls(start, goal, i)
        if result is not None:
            break
    return result


'''Iterative Deepening Search'''

def ids(start, goal):
    depth_limit = 0
    while True:
        result = dlss(start, goal, depth_limit)
        if result != None:
            return result
        depth_limit += 1


'''Greedy Search'''
class NodeH:
    def __init__(self, state, parent, action, cost, heuristic,weight):
        self.state = state
        self.cost = cost
        self.parent = parent
        self.action = action
        self.heuristic = heuristic
        self.weight = weight
        self.total_cost = self.cost + self.heuristic
        

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def getnewst(state, action):
    new_st = [row[:] for row in state]
    blank_i, blank_j = getblankloc(new_st)
    action_offset = {
        'Up': (-1, 0),
        'Down': (1, 0),
        'Left': (0, -1),
        'Right': (0, 1)
    }
    offset_i, offset_j = action_offset[action]
    new_st[blank_i][blank_j], new_st[blank_i+offset_i][blank_j+offset_j] = \
        new_st[blank_i+offset_i][blank_j+offset_j], new_st[blank_i][blank_j]
    return new_st

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def getheur(state, goal):
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    heuristic = 0
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val != goal[i][j]:
                goalpos = next((m, n) for m, row in enumerate(goal) for n, v in enumerate(row) if v == val)
                heuristic += manhattan_distance((i, j), goalpos)
    return heuristic


def getactionheru(state):
    blank_i, blank_j = getblankloc(state)
    row, col = len(state), len(state[0])
    return [action for action, (i, j) in [('Up', (blank_i-1, blank_j)), ('Down', (blank_i+1, blank_j)), 
                                          ('Left', (blank_i, blank_j-1)), ('Right', (blank_i, blank_j+1))] 
            if 0 <= i < row and 0 <= j < col]


def getpath(node):
    path = []
    while node:
        if node.action:
            path.append((node.action, node.weight))
        node = node.parent
    return list(reversed(path))
def getblankloc(state):
    for i, row in enumerate(state):
        if 0 in row:
            return (i, row.index(0))



def greedysearch(start, goal):
    nodespopped = 0
    nodesexpanded = 0
    maxfringesize = 0
    nodesgenerated = 1
    depth = 0
    cost = 0

    # Create root node
    root = NodeH(start, None, None, 0, getheur(start, goal), None)

    # Initialize fringe and visited set
    fringe = PriorityQueue()
    visited = set()

    # Add root node to fringe
    fringe.put(root)

    # Log initial state of fringe and visited set
    if dump == 'true':
        file = open(date_string, 'a')
        file.write(f"Generating successors to < state = {root.state}, action = {{Start}} g(n) = {root.cost}, d = 0, f(n) = {root.heuristic}, Parent = Pointer to {{None}} >:\n")
        file.write(f"Closed: {visited}\n")
        file.write(f"Fringe: [{root}]\n")

    while not fringe.empty():
        # Update max fringe size
        maxfringesize = max(maxfringesize, fringe.qsize())

        # Pop node from fringe
        node = fringe.get()
        nodespopped += 1

        # Check if goal state is reached
        if node.state == goal:
            # Update depth, cost, and steps
            depth = len(getpath(node))
            cost = node.cost
            steps = getpath(node)
            return nodespopped, nodesexpanded, nodesgenerated, maxfringesize, depth, cost, steps

        # Add current node to visited set
        visited.add(str(node.state))

        # Log state of current node and visited set
        if dump == 'true':
            file.write(f"Generating successors to < state = {node.state}:\n")
            file.write(f"\tClosed: {visited}\n")

        # Generate and add successors to fringe
        for action in getactionheru(node.state):
            # Generate new state based on action
            new_st = getnewst(node.state, action)

            # Calculate new cost
            new_cost = node.cost + new_st[getblankloc(node.state)[0]][getblankloc(node.state)[1]]

            # Check if new state has already been visited
            if str(new_st) not in visited:
                # Invert action for logging purposes
                if action == "Down":
                    action = "Up"
                elif action == "Up":
                    action = "Down"
                elif action == "Left":
                    action = "Right"
                else:
                    action = "Left"
                    
                # Create new node and add to fringe
                weight = new_st[getblankloc(node.state)[0]][getblankloc(node.state)[1]]
                new_node = NodeH(new_st, node, action, cost + new_cost, getheur(new_st, goal), weight)
                if new_node.heuristic <= 70:
                    nodesgenerated += 1
                    fringe.put(new_node)

                    # Log state of new node and fringe
                    if dump == 'true':
                        file.write(f"\tFringe: {new_node.state}\n")

        # Update nodes expanded
        nodesexpanded += 1

    return None



def astar(start, goal):
    # Initialize variables
    nodespopped, nodesexpanded, nodesgenerated, maxfringesize, depth, cost = 0, 0, 1, 0, 0, 0
    steps, visited = [], set()
    fringe = PriorityQueue()

    # Add start node to the fringe
    start_node = NodeH(start, None, None, 0, getheur(start, goal), None)
    fringe.put(start_node)

    # If dump is true, create a log file and write the first entry
    if dump:
        with open(date_string, 'a') as file:
            file.write(f"Generating successors to < state = {start_node.state}, action = {{Start}} g(n) = {start_node.cost}, d = 0, f(n) = {start_node.heuristic}, Parent = Pointer to {{None}} >:\n")
            file.write(f"\tClosed: {visited}\n")
            file.write(f"\tFringe: [{start_node}]\n")

    # While the fringe is not empty, keep searching
    while not fringe.empty():
        # Update the maximum fringe size seen so far
        maxfringesize = max(maxfringesize, fringe.qsize())

        # Get the next node from the fringe and increment the nodes popped counter
        node = fringe.get()
        nodespopped += 1

        # If the node is the goal state, construct the path and return results
        if node.state == goal:
            depth = len(getpath(node))
            cost = node.cost
            steps = getpath(node)
            return nodespopped, nodesexpanded, nodesgenerated, maxfringesize, depth, cost, steps

        # Add the node to the set of visited states
        visited.add(tuple(map(tuple, node.state)))

        # If dump is true, write a log entry for generating successors for the current node
        if dump:
            with open(date_string, 'a') as file:
                file.write(f"Generating successors to < state = {node.state}:\n")

        # Generate successors for the current node
        for action in getactionheru(node.state):
            new_st = getnewst(node.state, action)
            new_cost = node.cost + new_st[getblankloc(node.state)[0]][getblankloc(node.state)[1]]

            # If the new state has not been visited before, create a new node for it and add it to the fringe
            if tuple(map(tuple, new_st)) not in visited:
                if action == "Down":
                    action = "Up"
                elif action == "Up":
                    action = "Down"
                elif action == "Left":
                    action = "Right"
                else:
                    action = "Left"
                nodesgenerated += 1
                weight = new_st[getblankloc(node.state)[0]][getblankloc(node.state)[1]]
                new_node = NodeH(new_st, node, action, cost + new_cost, getheur(new_st, goal), weight)

                # If the new node's heuristic value is not too high, add it to the fringe
                if new_node.heuristic <= 50:
                    if dump:
                        with open(date_string, 'a') as file:
                            file.write(f"Closed: {visited}\n")
                            file.write(f"Fringe: {new_node.state}\n")
                    nodesgenerated += 1
                    fringe.put(new_node)

        # Increment the nodes expanded counter
        nodesexpanded += 1

    return "No solution found."


def changelist(data):
    return [data[i:i+3] for i in range(0, 9, 3)]


if __name__ == '__main__':
    if len(sys.argv) != 5:
        
        sys.exit()
    method = 'a*'
    dump = 'false'
    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    
    now = datetime.datetime.now()
    date_string = now.strftime("trace-%m_%d_%Y-%I_%M_%S_%p.txt")

    if (len(sys.argv) == 4):
        if sys.argv[3] == 'true' or sys.argv[3] == 'false':
            dump = sys.argv[3]
        else:
            method = sys.argv[3]
    if len(sys.argv) == 5:
        method = sys.argv[3]
        dump = sys.argv[4]

    # start = [[2, 3, 6], [1, 0, 7], [4, 8, 5]]
    # goal = [[1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 0]]
    # method = 'astar'
    # dump = "true"
    # search_algorithm = astar
    startfile = open(start_file, "r")
    goalfile = open(goal_file,"r")

    start = []
    goal = []

    with open(start_file, 'r') as startfile:
        for i in startfile:
            if i == 'END OF FILE':
                break
            i = i.replace("\n", '')
            i = i.split(" ")
            for j in i:
                start.append(int(j))

    with open(goal_file, 'r') as goalfile:
        for i in goalfile:
            if i == 'END OF FILE':
                break
            i = i.replace("\n", '')
            i = i.split(" ")
            for j in i:
                goal.append(int(j))

    if dump == 'true':
        file = open(date_string,'a')
        file.write(f"Commage-Line Arguments : ['{start_file}', '{goal_file}', '{method}', '{dump}']\n")
        file.write(f"Method seletion: {method}\n")
        file.write(f"Running {method}\n")
        file.close()
    

    if(method == "bfs"):
        result = bfs(tuple(start),tuple(goal))
    
    elif(method == "dfs"):
        result = dfs(start, goal)
    elif(method == "ucs"):
        start = changelist(start)
        goal = changelist(goal)
        result = ucs(start,goal)
    elif(method == "dls"):        
        depth_limit = int(input("Enter depth limit: "))
        result = dlss(start, goal,depth_limit)
    
    elif(method == "greedy"):
        start = changelist(start)
        goal = changelist(goal)
        result = greedysearch(start, goal)
    elif(method == "ids"):
        result = ids(start, goal)
    else:
        start = changelist(start)
        goal = changelist(goal)
        result = astar(start, goal)


    
    nodespopped, nodesexpanded, nodesgenerated, maxfringesize, depth, cost, steps = result
    
    print("Nodes Popped: {}".format(nodespopped))
    print("Nodes Expanded: {}".format(nodesexpanded))
    print("Nodes Genereated:{}".format(nodesgenerated))
    print("Max Fringe Size: {}".format(maxfringesize))
    print("Solution Found at depth", len(steps),"with cost of {}.".format(cost))
    print("Steps:")
    for move in steps:
        print("\tMove",move[1],move[0])

    if dump == 'true':        
        file = open(date_string,'a')
        file.write(f"\tNodes Popped: {nodespopped}\n")
        file.write(f"\tNodes expanded: {nodesexpanded}\n")
        file.write(f"\tNodes Genereated: {nodesgenerated}\n")
        file.write(f"\tMax Fringe Size: {maxfringesize}\n")
        





