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

from __future__ import division, print_function

from baselineTeam import DefensiveReflexAgent
from heapq import heappop, heappush, heapify, nsmallest
from operator import itemgetter
from captureAgents import CaptureAgent


#################
# Team creation #
#################
from mixins import InferenceMixin


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first='DijkstraMonteCarloAgent',
    second='DijkstraMonteCarloAgent',
):
    return [
        eval(first)(firstIndex, isRed, False),
        eval(second)(secondIndex, isRed, True)
    ]


################################################################################
# Concrete Agents
################################################################################
class DijkstraMonteCarloAgent(CaptureAgent, InferenceMixin):
    """
    This is a class define an offensive agent which use A* to initiate an
    optimal path to eat all food, and use Monte Carlo to escape from the chasers
    if the chaser is within the visible range
    """
    _instances = [None, None]
    _computed = False

    def __init__(self, index, red, defense, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        object.__init__(self)

        self.red = red
        self._defense = defense

        self._half = self._height = self._width = self._actions = None

        self._prevCarry = 0

        self._instances[index // 2] = self

    def registerInitialState(self, gameState):
        """
        Initialise the agent and compute an initial route
        """
        CaptureAgent.registerInitialState(self, gameState)

        data = gameState.data
        layout = data.layout
        self._height = layout.height
        width = self._width = layout.width
        self._half = width // 2

        self._initialise(gameState)

        if not self._defense:
            self._computeRoute(gameState)

    def _computeRoute(self, gameState):
        data = gameState.data
        foods = data.food.data
        height, width, half = self._height, self._width, self._half
        walls = data.layout.walls.data
        red = self.red
        bound = half - 1 if red else half
        bounds = set(
            (bound, y) for y in xrange(height) if not walls[bound][y]
        )
        foods = set(
            (x, y)
            for x in (xrange(half, width) if red else xrange(half))
            for y in xrange(height)
            if foods[x][y]
        )
        distancer = self.distancer

        pos = data.agentStates[self.index].configuration.pos
        path = []
        q = [(0, pos, path, foods)]
        while q:
            dist, pos, path, fs = heappop(q)

            if pos in bounds:
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
        path.reverse()

        self._actions = path

    def _notifyEaten(self, index, initPos):
        for ins in self._instances:
            ins._distribution[index] = {tuple(map(int, initPos)): 1.0}

    def _notifySeen(self, index, pos):
        # share visibility on the map
        for ins in self._instances:
            ins._distribution[index] = {tuple(map(int, pos)): 1.0}

    def _notifyDeath(self):
        # TODO: notify the teammate the death of the current agent
        pass

    def _roleChange(self):
        # TODO: return a boolean value to indicate if current role change
        pass

    def chooseAction(self, gameState):
        """
        Choose an action based on the current status of the agent
        """
        self._updateDistribution(gameState)

        return self.defenseAction(gameState) \
            if self._defense \
            else self.offenseAction(gameState)

    def defenseAction(self, gameState):
        """
        Choose a defensive action
        """
        index = self.index
        agentState = gameState.data.agentStates[index]
        pos = agentState.configuration.pos
        distancer = self.distancer

        # compute the expected distances to each opponent according to the
        # current distribution
        expected = [
            max(dist.items(), key=itemgetter(1))[0]
            for dist in self._distribution.values()
        ]
        closest = min(
            ((p, distancer.getDistance(pos, p)) for p in expected),
            key=itemgetter(1)
        )[0]
        a = [
            (a, gameState.generateSuccessor(index, a))
            for a in gameState.getLegalActions(index)
        ]
        a = [
            (a, successor)
            for a, successor in a
            if not successor.data.agentStates[index].isPacman
        ]
        a, _, s = min(
            (
                (
                    a,
                    distancer.getDistance(
                        closest,
                        successor.data.agentStates[index].configuration.pos
                    ),
                    successor
                )
                for a, successor in a
            ),
            key=itemgetter(1)
        )

        pos = s.data.agentStates[index].configuration.pos
        for i in (gameState.blueTeam if self.red else gameState.redTeam):
            conf = gameState.data.agentStates[i].configuration
            initPos = gameState.data.layout.agentPositions[i][1]
            if conf and conf.pos == pos:
                self._notifyEaten(i, initPos)

        return a

    def offenseAction(self, gameState):
        """
        Choose an offensive action based on current circumstance, could either
        be the followings
        * following the route
        * escape if in danger
        """
        index = self.index
        agentState = gameState.data.agentStates[index]

        if self._prevCarry > agentState.numCarrying:
            self._computeRoute(gameState)
        self._prevCarry = agentState.numCarrying

        distancer = self.distancer
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
