# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    class SearchNode:
        """
            Creates node: <state, action, parent_node>
        """
        def __init__(self, state, action=None, parent=None):
            self.state = state
            self.action = action
            self.parent = parent

        def extract_solution(self):
            """ Gets complete path from goal state to parent node """
            action_path = []
            search_node = self
            while search_node:
                if search_node.action:
                    action_path.append(search_node.action)
                search_node = search_node.parent
            return list(reversed(action_path))

    start_node = SearchNode(problem.getStartState())

    if problem.isGoalState(start_node.state):
        return start_node.extract_solution()

    frontier = util.Stack()
    explored = set()
    frontier.push(start_node)

    # run until stack is empty
    while not frontier.isEmpty():
        node = frontier.pop()  # choose the deepest node in frontier
        explored.add(node.state)

        if problem.isGoalState(node.state):
            return node.extract_solution()

        # expand node
        successors = problem.getSuccessors(node.state)

        for succ in successors:
            # make-child-node
            child_node = SearchNode(succ[0], succ[1], node)
            if child_node.state not in explored:
                frontier.push(child_node)

    # no solution
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    class SearchNode:
        """
            Creates node: <state, action, parent_node>
        """
        def __init__(self, state, action=None, parent=None):
            self.state = state
            self.action = action
            self.parent = parent

        def extract_solution(self):
            """ Gets complete path from goal state to parent node """
            action_path = []
            search_node = self
            while search_node:
                if search_node.action:
                    action_path.append(search_node.action)
                search_node = search_node.parent
            return list(reversed(action_path))

        def is_in_frontier(self, data_structure):
            for n in data_structure.list:
                if n.state == self.state:
                    return True
            return False


    start_node = SearchNode(problem.getStartState())

    if problem.isGoalState(start_node.state):
        return start_node.extract_solution()

    frontier = util.Queue() # FIFO
    frontier.push(start_node)
    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()  # choose the shallowest node in frontier
        explored.add(node.state)

        if problem.isGoalState(node.state):
            return node.extract_solution()

        successors = problem.getSuccessors(node.state)
        for succ in successors:
            child_node = SearchNode(succ[0], succ[1], node)
            if child_node.state not in explored and\
                not child_node.is_in_frontier(frontier):
                frontier.push(child_node)

    # no solution
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    class SearchNode:
        """
            Creates node: <state, action, cost, parent_node>
        """
        def __init__(self, state, action=None, path_cost = 0, parent=None):
            self.state = state
            self.action = action
            self.parent = parent
            if parent:
                self.path_cost = path_cost + parent.path_cost
            else:
                self.path_cost = path_cost

        def extract_solution(self):
            """ Gets complete path from goal state to parent node """
            action_path = []
            search_node = self
            while search_node:
                if search_node.action:
                    action_path.append(search_node.action)
                search_node = search_node.parent
            return list(reversed(action_path))

        def is_in_priority_queue(self, priority_queue):
            """ Check if the node is already in the priority queue """
            for index, (p, c, i) in enumerate(priority_queue.heap):
                if i.state == self.state:
                    return True
            else:
                return False

    start_node = SearchNode(problem.getStartState())

    if problem.isGoalState(start_node.state):
        return start_node.extract_solution()

    frontier = util.PriorityQueue()  # FIFO
    frontier.push(start_node, start_node.path_cost)
    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()  # chooses the lowest-cost node in frontier

        # goal-test
        if problem.isGoalState(node.state):
            return node.extract_solution()

        if node.state not in explored:
            explored.add(node.state)

            successors = problem.getSuccessors(node.state)

            for succ in successors:
                child_node = SearchNode(succ[0], succ[1], succ[2], node)
                frontier.update(child_node, child_node.path_cost)

    # no solution
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


# In both pratical task and Assignment 1
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # class to represent SearchNode
    class SearchNode:
        """
            Creates node: <state, action, f(s), g(s), h(s), parent_node>
        """
        def __init__(self, state, action=None, g=None, h=None,
                     parent=None):
            self.state = state
            self.action = action
            self.parent = parent
            # heuristic value
            self.h = h
            # combined cost
            if parent:
                self.g = g + parent.g
            else:
                self.g = 0
            # evaluation function value
            self.f = self.g + self.h

        def extract_solution(self):
            """ Gets complete path from goal state to parent node """
            action_path = []
            search_node = self
            while search_node:
                if search_node.action:
                    action_path.append(search_node.action)
                search_node = search_node.parent
            return list(reversed(action_path))


    # make search node function
    def make_search_node(state, action=None, cost=None, parent=None):
        if hasattr(problem, 'heuristicInfo'):
            if parent:
                # same parent - avoid re-calculation
                # for reducing computations in logic
                if parent == problem.heuristicInfo["parent"]:
                    problem.heuristicInfo["sameParent"] = True
                else:
                    problem.heuristicInfo["sameParent"] = False
            # adding parent info for reducing computations
            problem.heuristicInfo["parent"] = parent
        # get heuristic value
        h_value = heuristic(state, problem)
        return SearchNode(state, action, cost, h_value, parent)

    # create open list
    open = util.PriorityQueue()
    node = make_search_node(problem.getStartState())
    open.push(node, node.f)
    closed = set()
    best_g = {}  # maps states to numbers

    # run until open list is empty
    while not open.isEmpty():
        node = open.pop()  # pop-min

        if node.state not in closed or node.g < best_g[node.state]:
            closed.add(node.state)
            best_g[node.state] = node.g

            # goal-test
            if problem.isGoalState(node.state):
                return node.extract_solution()

            # expand node
            successors = problem.getSuccessors(node.state)
            for succ in successors:
                child_node = make_search_node(succ[0],succ[1],succ[2], node)
                if child_node.h < float("inf"):
                    open.push(child_node, child_node.f)

    # no solution
    util.raiseNotDefined()

# Extensions Assignment 1
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""

    class SearchNode:
        """
            Creates node: <state, action, depth, parent_node>
        """
        def __init__(self, state, action=None, depth = 0, parent=None):
            self.state = state
            self.action = action
            self.parent = parent
            if parent:
                self.depth = depth + parent.depth
            else:
                self.depth = depth

        def extract_solution(self):
            """ Gets complete path from initial state to goal state """
            action_path = []
            search_node = self
            while search_node:
                if search_node.action:
                    action_path.append(search_node.action)
                search_node = search_node.parent
            return list(reversed(action_path))

    # limit for IDS
    limit = 0

    # controlling infinite loop
    LOOP_COUNT = 0
    LOOP_LIMIT = 999999999

    # running iteratively
    # increasing limit until goal-state is found
    while True:

        # no solution hard limit check
        if LOOP_COUNT == LOOP_LIMIT:
            break

        node = SearchNode(problem.getStartState())

        # goal-test
        if problem.isGoalState(node.state):
            return node.extract_solution()

        frontier = util.Stack()     # LIFO stack
        explored = set()            # empty set
        frontier.push(node)

        # run until frontier is empty
        while not frontier.isEmpty():
            node = frontier.pop()  # choose the deepest node in frontier
            explored.add(node.state)

            # never expand branch farther than the limit
            if node.depth < limit:
                # expand node
                successors = problem.getSuccessors(node.state)

                for succ in successors:
                    # make-child-node
                    # path step cost is considered as depth
                    child_node = SearchNode(succ[0], succ[1], succ[2], node)
                    # child.STATE is not in explored
                    if child_node.state not in explored:
                        # GOAL-TEST done on generation
                        if problem.isGoalState(child_node.state):
                            return child_node.extract_solution()
                        frontier.push(child_node)

        # goal-state not found -> increase limit by 1
        limit += 1
        LOOP_COUNT += 1

    # no solution
    util.raiseNotDefined()


def enforcedHillClimbing(problem, heuristic=nullHeuristic):
    """
    Local search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call it.
    The heuristic function is "manhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second arguement (heuristic).
    """
    # class to represent SearchNode
    class SearchNode:
        """
            Creates node: <state, action, h(s), parent_node>
        """
        def __init__(self, state, action=None, h = None, parent=None):
            self.state = state
            self.action = action
            self.parent = parent
            self.h = h

        def extract_solution(self):
            """ Gets complete path from goal state to parent node """
            action_path = []
            search_node = self
            while search_node:
                if search_node.action:
                    action_path.append(search_node.action)
                search_node = search_node.parent
            return list(reversed(action_path))

    # make search node function
    def make_search_node(state, action = None, parent = None):
        h_value = heuristic(state, problem)
        return SearchNode(state, action, h_value, parent)

    # improve helper function
    def improve(node_to_improve):

        queue = util.Queue()  # FIFO queue
        queue.push(node_to_improve)
        closed = set()

        while not queue.isEmpty():
            node = queue.pop()  # pop-front
            if node.state not in closed:
                closed.add(node.state)

                if node.h < node_to_improve.h:
                    return node

                successors = problem.getSuccessors(node.state)
                for succ in successors:
                    new_node = make_search_node(succ[0], succ[1], node)
                    queue.push(new_node)
        # fail
        return None

    # main iterative loop
    node = make_search_node(problem.getStartState())
    while not problem.isGoalState(node.state):
        node = improve(node)

    if node:
        return node.extract_solution()
    else:
        # no solution
        util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
ehc = enforcedHillClimbing
