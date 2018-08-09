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

from operator import attrgetter
from util import *

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
        raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


class NodeGenerator(object):
    __slots__ = ()

    def __call__(self, data, prev=None, prevDir=None)
        return data, prev, prevDir


class CostNodeGenerator(NodeGenerator):
    __slots__ = ()

    def __call__(self, g, cost, data, prev=None, prevDir=None):
        return super(CostNodeGenerator, self).__call__(data, prev, prevDir), \
               g + cost


class HeuristicNodeGenerator(CostNodeGenerator):
    __slots__ = "heuristic"

    def __init__(self, heuristic):
        self.heuristic = heuristic

    def __call__(self, g, cost, data, prev=None, prevDir=None):
        node, g = super(HeuristicNodeGenerator, self).__call__(
            g, cost, data, prev, prevDir
        )
        return node, g, self.heuristic()


def simpleSearch(problem, Container):
    container = Container()
    container.push(SearchNode(problem.getStartState()))
    visited = {}
    goal = False

    while not container.isEmpty():
        n = container.pop()
        # test if goal achieve
        if problem.isGoalState(n.data):
            goal = True
            break
        if n.data in visited:
            continue
        # add to close set
        visited[n.data] = n

        succs = problem.getSuccessors(n.data)
        # expand unvisited child nodes
        for succ, d, cost in succs:
            if succ not in visited:
                container.push(SearchNode(succ, n, d, cost))

    path = []
    if goal:
        while n.prevNode:
            path.append(n.prevDir)
            n = n.prevNode
        path.reverse()

    return path


def depthFirstSearch(problem):
    return simpleSearch(problem, Stack)


def breadthFirstSearch(problem):
    return simpleSearch(problem, Queue)


def prioritySearch(problem, Node, prioritise):
    q = PriorityQueue()
    n = Node(problem.getStartState())
    q.push(n, prioritise(n))
    visited = {}
    goal = False

    while not q.isEmpty():
        n = q.pop()
        # test if goal achieve
        if problem.isGoalState(n.data):
            goal = True
            break
        if n.data in visited:
            continue
        # add to close set
        visited[n.data] = n

        succs = problem.getSuccessors(n.data)
        # expand unvisited child nodes
        for succ, d, cost in succs:
            if succ in visited:
                succ = visited[succ]
            else:
                n = Node(succ, n, d, cost)
                q.push(n, prioritise(n))

    path = []
    if goal:
        while n.prevNode:
            path.append(n.prevDir)
            n = n.prevNode
        path.reverse()

    return path


def uniformCostSearch(problem):
    return prioritySearch(problem, NodeGenerator(), attrgetter("g"))


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
