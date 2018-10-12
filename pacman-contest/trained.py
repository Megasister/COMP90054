# agents.py
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

################################################################################
# Imports
################################################################################
from __future__ import division, print_function
from baselineTeam import DefensiveReflexAgent
from heapq import heappop, heappush, heapify, nsmallest
from operator import itemgetter, add
from captureAgents import CaptureAgent
import numpy as np
import numpy.random as npr

################################################################################
# Constants
################################################################################
TRAIN = True
ETA = 0.1


#################
# Team creation #
#################
from distanceCalculator import manhattanDistance
from game import Directions
from mixins import InferenceMixin


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


################################################################################
# Helper Function
################################################################################
def _applyKernel(accum, kernel):
    weight, bias = kernel
    return accum.dot(weight) + bias


################################################################################
# Agent
################################################################################
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

        lc = self._lc = 3
        fc = self._fc = 66

        try:
            self._omodel = [
                (np.load("oweight%d.py" % i), np.load("obias%d.py" % i))
                for i in range(lc)
            ]
            self._dmodel = [
                (np.load("dweight%d.py" % i), np.load("dbias%d.py" % i))
                for i in range(lc)
            ]
            self._fmodel = [
                (np.load("fweight%d.py" % i), np.load("fbias%d.py" % i))
                for i in range(lc)
            ]
        except:
            factor1 = npr.randint(-11, 10, (5 * fc, fc))
            factor2 = npr.randint(-11, 10, (fc, fc))
            factor3 = npr.randint(-11, 10, (fc, 4))
            self._omodel = [
                (
                    npr.random((5 * fc, fc)) + 1 + factor1,
                    npr.random((5 * fc, fc)) + factor1
                ),
                (
                    npr.random((fc, fc)) + 1 + factor2,
                    npr.random((fc, fc)) + factor2
                ),
                (
                    npr.random((fc, 5)) + 1 + factor3,
                    npr.random((fc, 5)) + factor3
                )
            ]
            self._dmodel = [
                (
                    npr.random((5 * fc, fc)) + 1 + factor1,
                    npr.random((5 * fc, fc)) + factor1
                ),
                (
                    npr.random((fc, fc)) + 1 + factor2,
                    npr.random((fc, fc)) + factor2
                ),
                (
                    npr.random((fc, 5)) + 1 + factor3,
                    npr.random((fc, 5)) + factor3
                )
            ]
            factor4 = npr.randint(-2, 1, (8, 8))
            factor5 = npr.randint(-2, 1, (8, 4))
            self._fmodel = [
                (
                    npr.random((8, 8)) + 1 + factor4,
                    npr.random((8, 8)) + factor4
                ),
                (
                    npr.random((8, 4)) + 1 + factor5,
                    npr.random((8, 2)) + factor5
                )
            ]

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

    def final(self, gameState):
        """
        Receive final game state
        """
        CaptureAgent.final(self, gameState)

        if TRAIN:
            for i, (weight, bias) in enumerate(self._omodel):
                np.save("oweight%d" % i, weight)
                np.save("obias%d" % i, bias)
            for i, (weight, bias) in enumerate(self._dmodel):
                np.save("dweight%d" % i, weight)
                np.save("dbias%d" % i, bias)
            for i, (weight, bias) in enumerate(self._fmodel):
                np.save("fweight%d" % i, weight)
                np.save("fbias%d" % i, bias)

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

    def _extractNear(self, pos, layout, agentStates, oppo):
        height = layout.height
        width = layout.width
        half = width // 2
        print(layout, height, width)
        x, y = pos
        lx, ly, rx, ry = x - 5, y - 5, x + 5, y + 5
        print(lx, ly, rx, ry)
        blx = lx if lx > 0 else 0
        bly = ly if ly > 0 else 0
        brx = rx if rx < width else width - 1
        bry = ry if ry < height else height - 1
        print(blx, bly, brx, bry)
        lx = blx - lx
        ly = bly - ly
        rx -= brx
        ry -= bry
        print(lx, ly, rx, ry)

        foods = np.array(layout.food.data, np.bool)
        print(foods)
        tmp = foods.copy()
        tmp[:, :half] = False
        print(tmp)
        tfood = np.zeros((11, 11), np.bool)
        tfood[ly:-ry, lx:-rx] = tmp[bly:-bry, blx:-brx]
        print(tfood)
        tmp = foods.copy()
        tmp[:, half:] = False
        print(tmp)
        ofood = np.zeros((11, 11), np.bool)
        ofood[ly:-ry, lx:-rx] = tmp[bly:-bry, blx:-brx]
        print(ofood)

        capsules = np.zeros_like(foods, np.bool)
        capsules[zip(*layout.capsules)] = True
        tmp = capsules.copy()
        tmp[:, :half] = False
        print(tmp)
        tcapsule = np.zeros((11, 11), np.bool)
        tcapsule[ly:-ry, lx:-rx] = tmp[bly:-bry, blx:-brx]
        print(tcapsule)
        tmp = capsules.copy()
        tmp[:, half:] = False
        print(tmp)
        ocapsule = np.zeros((11, 11), np.bool)
        ocapsule[ly:-ry, lx:-rx] = tmp[bly:-bry, blx:-brx]
        print(ocapsule)

        agentPos = [agentStates[i].configuration.pos for i in oppo]
        enemies = np.zeros_like(foods, np.bool)
        enemies[zip(*agentPos)] = True
        tmp = enemies.copy()
        tmp[:, :half] = False
        print(tmp)
        tcapsule = np.zeros((11, 11), np.bool)
        tcapsule[ly:-ry, lx:-rx] = tmp[bly:-bry, blx:-brx]
        print(tcapsule)
        tmp = capsules.copy()
        tmp[:, half:] = False
        print(tmp)
        ocapsule = np.zeros((11, 11), np.bool)
        ocapsule[ly:-ry, lx:-rx] = tmp[bly:-bry, blx:-brx]
        print(ocapsule)

        exit()

    def _extractOffensive(self, gameState):
        fs = np.zeros((1, self._fc), np.float64)

        red = self.red
        if red:
            team = gameState.redTeam
            oppo = gameState.blueTeam
        else:
            team = gameState.blueTeam
            oppo = gameState.redTeam
        data = gameState.data
        agentStates = data.agentStates
        agent = agentStates[self.index]
        pos = agent.configuration.pos

        fs[0] = sum(agentStates[i].scaredTimer > 0 for i in oppo)

        return fs

    def _extractDefensive(self, gameState):
        fs = np.zeros((1, self._fc), np.float64)

        red = self.red
        if red:
            team = gameState.redTeam
            oppo = gameState.blueTeam
        else:
            team = gameState.blueTeam
            oppo = gameState.redTeam
        data = gameState.data
        agentStates = data.agentStates
        agent = agentStates[self.index]
        pos = agent.configuration.pos

        fs[0] = sum(agentStates[i].scaredTimer > 0 for i in team)

        return fs

    def _predict(self, gameState, defensive):
        if defensive:
            model = self._dmodel
            fs = self._extractDefensive(gameState)
        else:
            model = self._omodel
            fs = self._extractOffensive(gameState)

        return reduce(_applyKernel, model, fs)

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
            return MCTS(index, gameState, self._evalOffense)

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
