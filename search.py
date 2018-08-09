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
    return [s, s, w, s, w, w, s, w]


def simpleSearch(problem, Container):
    container = Container()
    container.push((problem.getStartState(), None, None))
    visited = {}
    goal = False

    while not container.isEmpty():
        n = container.pop()
        # test if goal achieve
        if problem.isGoalState(n[0]):
            goal = True
            break
        if n[0] in visited:
            continue
        # add to close set
        visited[n[0]] = n

        succs = problem.getSuccessors(n[0])
        # expand unvisited child nodes
        for succ, d, _ in succs:
            if succ not in visited:
                container.push((succ, n[0], d))

    path = []
    if goal:
        while n[1]:
            path.append(n[2])
            n = visited[n[1]]
        path.reverse()

    return path


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return simpleSearch(problem, util.Stack)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return simpleSearch(problem, util.Queue)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    q = util.PriorityQueue()
    n = problem.getStartState()
    q.push(n, 0)
    visited = {n: (None, None, 0)}
    goal = False

    while not q.isEmpty():
        n = q.pop()
        if problem.isGoalState(n):
            goal = True
            break

        for succ, d, cost in problem.getSuccessors(n):
            dist = visited[n][2] + cost
            # expand the node if it is not visited yet or it has a lower
            # distance from the original node
            if succ not in visited or dist < visited[succ][2]:
                q.push(succ, dist)
                visited[succ] = (n, d, dist)

    path = []
    if goal:
        n, d, _ = visited[n]
        while d is not None:
            path.append(d)
            n, d, _ = visited[n]
    path.reverse()

    return path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    q = util.PriorityQueue()
    n = problem.getStartState()
    q.push(n, 0)
    h = heuristic(n, problem)
    visited = {n: (None, None, (h, 0, h))}
    goal = False

    while not q.isEmpty():
        n = q.pop()
        if problem.isGoalState(n):
            goal = True
            break

        for succ, d, cost in problem.getSuccessors(n):
            dist = visited[n][2][1] + cost
            # expand the node if it is not visited yet or it has a lower
            # distance from the original node
            if succ not in visited or dist < visited[succ][2][1]:
                h = heuristic(succ, problem)
                f = dist + h
                q.push(succ, (f, h))
                visited[succ] = (n, d, (f, dist, h))

    path = []
    if goal:
        n, d, _ = visited[n]
        while d is not None:
            path.append(d)
            n, d, _ = visited[n]
    path.reverse()

    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
