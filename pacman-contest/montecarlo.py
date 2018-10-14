# montecarlo.py
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

from __future__ import division, print_function

import random
from heapq import heappop, heappush
from operator import add, itemgetter

from captureAgents import CaptureAgent


#################
# Team creation #
#################
from distanceCalculator import manhattanDistance
from game import Directions


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first='AbuseMonteCarloAgent',
    second='AbuseMonteCarloAgent',
):
    return [
        eval(first)(firstIndex, isRed, False),
        eval(second)(secondIndex, isRed, True)
    ]


# helper function to perform element-wise addition on two lists
def _sum_list(x, y):
    return map(add, x, y)


################################################################################
# Mixed Agents
################################################################################
def MonteCarlo(index, gameState, evaluate, depth=40, count=100):
    """
    Monte Carlo Tree Search
    """
    actions = gameState.getLegalActions(index)
    actions.remove(Directions.STOP)
    wins = [0] * len(actions)
    for i, action in enumerate(actions):
        for c in xrange(count):
            ind, d = index, depth
            prevScore = -float("inf")
            successor = gameState.generateSuccessor(ind, action)
            score = evaluate(successor)
            while score >= prevScore and d > 1:
                prevScore = score
                ind = (ind + 1) % 4
                act = successor.getLegalActions(ind)
                act.remove(Directions.STOP)
                successor = successor.generateSuccessor(
                    ind, random.choice(act)
                )
                score = evaluate(successor)
                d -= 1
            wins[i] += score > prevScore
    ms = max(wins)
    return random.choice([a for a, s in zip(actions, wins) if s == ms])


class AbuseMonteCarloAgent(CaptureAgent, object):
    """
    This is a class define an offensive agent which use A* to initiate an
    optimal path to eat all food, and use Monte Carlo to escape from the chasers
    if the chaser is within the visible range
    """

    _instances = [None, None]

    def __init__(self, index, red, defense, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        object.__init__(self)

        self.red = red
        self._defense = defense
        self._height = self._width = self._half = self._bound = \
            self._actions = None
        self._prevCarry = 0
        self._recompute = False

        # record each instance created
        self._instances[index // 2] = self

    def registerInitialState(self, gameState):
        """
        Initialise the agent and compute an initial route
        """
        CaptureAgent.registerInitialState(self, gameState)

        data = gameState.data
        layout = data.layout
        height = self._height = layout.height
        width = self._width = layout.width
        self._half = half = width // 2
        red = self.red
        bound = half - 1 if red else half
        walls = layout.walls
        self._bound = set(
            (bound, y) for y in xrange(height) if not walls[bound][y]
        )

        # only offensive agent needs to compute the route
        if not self._defense:
            self._computeRoute(gameState)

    def _computeRoute(self, gameState):
        data = gameState.data
        foods = data.food.data
        height, width, half = self._height, self._width, self._half
        red = self.red
        # deliver the food to the bound to gain actual score
        bounds = self._bound
        foods = set(
            (x, y)
            for x in (xrange(half, width) if red else xrange(half))
            for y in xrange(height)
            if foods[x][y]
        )
        distancer = self.distancer

        # Dijkstra (or variant of Uniform Cost Search) implementation
        pos = data.agentStates[self.index].configuration.pos
        path = []
        q = [(0, pos, path, foods)]
        while q:
            dist, pos, path, fs = heappop(q)

            if pos in bounds and not fs:
                break

            if fs:
                npos, ndist = min(
                    ((np, dist + distancer.getDistance(pos, np)) for np in fs),
                    key=itemgetter(1)
                )

                nfs = fs.copy()
                nfs.remove(npos)
                heappush(q, (ndist, npos, path + [npos], nfs))
            else:
                npos, ndist = min(
                    (
                        (np, dist + distancer.getDistance(pos, np))
                        for np in bounds
                    ),
                    key=itemgetter(1)
                )
                heappush(q, (ndist, npos, path + [npos], None))
        # reverse the path to utilise the efficiency of list.pop
        path.reverse()

        self._actions = path

    def observationFunction(self, gameState):
        """
        Abuse this function to obtain a full game state from the controller
        """
        return gameState

    def chooseAction(self, gameState):
        """
        Choose an action based on the current status of the agent
        """

        return self.defenseAction(gameState) \
            if self._defense \
            else self.offenseAction(gameState)

    def defenseAction(self, gameState):
        """
        Choose a defensive action
        """
        index = self.index
        red = self.red
        data = gameState.data
        agentStates = data.agentStates
        agentState = agentStates[index]
        pos = agentState.configuration.pos
        distancer = self.distancer

        # obtain the states of opponent agents
        states = [
            agentStates[i]
            for i in (gameState.blueTeam if red else gameState.redTeam)
        ]
        vals = [(s.numCarrying, s.configuration.pos) for s in states]
        p = min(((-c, distancer.getDistance(pos, p), p) for c, p in vals))[2]
        a = [
            (a, gameState.generateSuccessor(index, a))
            for a in gameState.getLegalActions(index)
        ]
        a = [
            (a, successor)
            for a, successor in a
            if not successor.data.agentStates[index].isPacman
        ]

        return min(
            (
                (
                    a, distancer.getDistance(
                        successor.data.agentStates[index].configuration.pos,
                        p
                    )
                )
                for a, successor in a
            ),
            key=itemgetter(1)
        )[0]

    def _evalOffense(self, gameState):
        fs = [0] * 6

        data = gameState.data
        agentStates = data.agentStates
        agent = agentStates[self.index]
        pos = agent.configuration.pos
        maxDist = self._height * self._width
        distancer = self.distancer
        red = self.red

        # current score
        fs[0] = data.score

        for i in gameState.redTeam:
            agentState = agentStates[i]
            if red:
                fs[1] += agentState.numCarrying
            else:
                if agent.isPacman and agentState.scaredTimer == 0:
                    fs[2] += distancer.getDistance(
                        pos, agentState.configuration.pos
                    )
                if agentState.scaredTimer > 0:
                    fs[4] += 1
        for i in gameState.blueTeam:
            agentState = agentStates[i]
            if red:
                if agent.isPacman and agentState.scaredTimer == 0:
                    fs[2] += distancer.getDistance(
                        pos, agentState.configuration.pos
                    )
                if agentState.scaredTimer > 0:
                    fs[4] += 1
            else:
                fs[1] += agentState.numCarrying

        fs[3] = len(gameState.getLegalActions(self.index))

        fs[5] = agent.isPacman

        weights = [maxDist * 2, 2, 2, 1, 1, maxDist]
        return sum(f * w for f, w in zip(fs, weights))

    def _evalDefense(self, gameState):
        fs = [0] * 5

        data = gameState.data
        agentStates = data.agentStates
        agent = agentStates[self.index]
        pos = agent.configuration.pos
        distancer = self.distancer
        red = self.red

        for i in gameState.redTeam:
            agentState = agentStates[i]
            if red:
                if agentState.isPacman and agent.scaredTimer == 0:
                    fs[2] += distancer.getDistance(
                        pos, agentState.configuration.pos
                    )
            else:
                fs[1] += agentState.numCarrying
                fs[3] = len(gameState.getLegalActions(i))
        for i in gameState.blueTeam:
            agentState = agentStates[i]
            if red:
                fs[1] += agentState.numCarrying
                fs[3] = len(gameState.getLegalActions(i))
            else:
                if agentState.isPacman and agent.scaredTimer == 0:
                    fs[2] += distancer.getDistance(
                        pos, agentState.configuration.pos
                    )

        weights = [200, 2, 2, 1, 1]
        return sum(f * w for f, w in zip(fs, weights))

    def offenseAction(self, gameState):
        """
        Choose an offensive action based on current circumstance, could either
        be the followings
        * following the route
        * escape if in danger
        """
        index = self.index
        red = self.red
        agentStates = gameState.data.agentStates
        agentState = agentStates[index]

        distancer = self.distancer
        pos = agentState.configuration.pos
        states = [
            agentStates[i]
            for i in (gameState.blueTeam if red else gameState.redTeam)
        ]
        if any(
            not s.isPacman and s.scaredTimer == 0 and distancer.getDistance(
                s.configuration.pos, pos
            ) < 6
            for s in states
        ):
            self._recompute = True
            self._prevCarry = agentState.numCarrying
            return MonteCarlo(index, gameState, self._evalOffense)

        if self._recompute or self._prevCarry > agentState.numCarrying:
            self._computeRoute(gameState)
            self._recompute = False
        self._prevCarry = agentState.numCarrying

        actions = self._actions
        if agentState.configuration.pos == actions[-1]:
            actions.pop()
        cdest = actions[-1]
        return min(
            (
                (
                    a,
                    distancer.getDistance(gameState.generateSuccessor(
                        index, a
                    ).data.agentStates[index].configuration.pos, cdest)
                )
                for a in gameState.getLegalActions(index)
            ),
            key=itemgetter(1)
        )[0]
