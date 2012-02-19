# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

class Node:
    """
    Represents a node in a search graph, consisting of a state,
    parent-node, action applied to parent, path-cost, and depth.
    """
    def __init__(self, state, parent, action, pathCost, depth):
        self.state = state
        self.parent = parent
        self.action = action
        self.pathCost = pathCost
        self.depth = depth

def expand(node, problem):
    """
    Returns a list of successor nodes.
    """
    successors = []
    for successor, action, stepCost in problem.getSuccessors(node.state):
        s = Node(successor, node, action, node.pathCost + stepCost, node.depth + 1)
        successors.append(s)
    return successors

def solution(node):
    """
    Returns a list of actions, following parent pointers.
    """
    if node.parent == None:
        return []
    ls = solution(node.parent)
    ls.extend([node.action])
    return ls

def graphSearch(problem, fringe):
    """
    Generic graph-search that takes in a problem and a fringe
    (which dictates what type of traversal this is), to return
    a list of actions that reaches the goal.
    [2nd Edition: Figure 3.19]
    """
    closed = set([])
    fringe.push(Node(problem.getStartState(), None, None, 0, 0))
    while(True):
        if fringe.isEmpty():
            return False
        node = fringe.pop()
        if problem.isGoalState(node.state):
            return solution(node)
        if node.state not in closed:
            closed.add(node.state)
            for newNode in expand(node, problem):
                fringe.push(newNode)

def depthFirstSearch(problem):
    from util import Stack
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]
    """
    return graphSearch(problem, Stack())

def breadthFirstSearch(problem):
    from util import Queue
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    return graphSearch(problem, Queue())

def uniformCostSearch(problem):
    from util import PriorityQueueWithFunction
    "Search the node of least total cost first. "
    return graphSearch(problem, PriorityQueueWithFunction(lambda item: item.pathCost))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueueWithFunction
    "Search the node that has the lowest combined cost and heuristic first."
    return graphSearch(problem, PriorityQueueWithFunction(lambda item: item.pathCost + heuristic(item.state, problem)))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
