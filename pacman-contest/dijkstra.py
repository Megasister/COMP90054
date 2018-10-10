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

import random

from baselineTeam import DefensiveReflexAgent
from heapq import heappop, heappush, heapify, nsmallest
from operator import itemgetter, add
from captureAgents import CaptureAgent


#################
# Team creation #
#################
from distanceCalculator import manhattanDistance
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


# helper function to perform element-wise addition on two lists
def _sum_list(x, y):
    return map(add, x, y)


################################################################################
# Mixed Agents
################################################################################
class DijkstraMonteCarloAgent(CaptureAgent, object):
    """
    This is a class define an offensive agent which use A* to initiate an
    optimal path to eat all food, and use Monte Carlo to escape from the chasers
    if the chaser is within the visible range
    """

    _DIRECTION = (
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0)
    )

    _distribution = _validPos = None
    _initialised = False

    @classmethod
    def _validMove(cls, pos, wall):
        x, y = pos
        # a STOP is possible
        ds = cls._DIRECTION
        return [pos] + [
            (x + dx, y + dy)
            for dx, dy in ds
            if not wall[x + dx][y + dy]
        ]

    _instances = [None, None]
    _computed = False

    def __init__(self, index, red, defense, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        object.__init__(self)

        self.red = red
        self._defense = defense
        self._bound = self._actions = None
        self._prevCarry = 0

        # record each instance created
        self._instances[index // 2] = self
        self._teammate = (index // 2 + 1) % 2

    def registerInitialState(self, gameState):
        """
        Initialise the agent and compute an initial route
        """
        CaptureAgent.registerInitialState(self, gameState)

        data = gameState.data
        layout = data.layout
        height, width = layout.height, layout.width
        half = width // 2
        red = self.red
        bound = half - 1 if red else half
        walls = layout.walls
        self._bound = set(
            (bound, y) for y in xrange(height) if not walls[bound][y]
        )

        self._initialise(gameState, red)

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
        # reverse the path to utilise the efficiency of list.pop
        path.reverse()

        self._actions = path

    @classmethod
    def _initialise(cls, initState, red):
        if cls._initialised:
            return

        # initialise all valid positions with the given wall configuration
        data = initState.data
        wall = data.layout.walls
        height, width = wall.height, wall.width
        wall = data.layout.walls.data
        cls._validPos = [
            (x, y)
            for x in xrange(width)
            for y in xrange(height)
            if not wall[x][y]
        ]

        # the initial position is visible
        agentStates = data.agentStates
        cls._distribution = {
            i: {tuple(map(int, agentStates[i].configuration.pos)): 1.0}
            for i in (initState.blueTeam if red else initState.redTeam)
        }

        cls._initialised = True

    def _notifyDeath(self):
        # TODO: notify the teammate the death of the current agent
        pass

    def _roleChange(self):
        # TODO: return a boolean value to indicate if current role change
        pass

    def _nextDistribution(self, gameState):
        agentDistances = gameState.agentDistances
        data = gameState.data
        layout = data.layout
        wall = layout.walls.data
        agentStates = data.agentStates
        vp = self._validPos
        dist = {}
        pos = agentStates[self.index].configuration.pos
        print("--------", self._distribution)

        for agent, adist in self._distribution.items():
            agentState = agentStates[agent]
            conf = agentState.configuration
            # if the agent is visible
            if conf is not None:
                pos = tuple(map(int, conf.pos))
                dist[agent] = {pos: 1.0}
                self._instances[self._teammate]._distribution[agent] = {
                    pos: 1.0
                }
                continue

            ndist = agentDistances[agent]
            print("--------", ndist)
            nd = {}
            # find all valid positions according to the noisy distance
            ps = [p for p in vp if -7 < manhattanDistance(pos, p) - ndist < 7]
            # iterate over all existing positions
            for k, v in adist.items():
                # generate all valid positions after possible move
                vm = self._validMove(k, wall)
                # count the number of valid points for each of the possible
                # position after movement
                cp = reduce(_sum_list, ([
                    manhattanDistance(m, p) < 7 for m in vm
                ] for p in ps))
                scp = sum(cp)
                # normalise to get a vector of probabilities to redistribute the
                # probability
                if scp != 0:
                    cp = [c / scp for c in cp]
                for m, p in zip(vm, cp):
                    nd[m] = nd.get(m, 0) + v * p

            # remove all zero probability and normalise the probability
            tp = sum(nd.values())
            dist[agent] = {k: v / tp for k, v in nd.items() if v > 0}
        self._distribution = dist

    def chooseAction(self, gameState):
        """
        Choose an action based on the current status of the agent
        """
        self._nextDistribution(gameState)
        print("====", self.index)
        print(self._distribution)

        return self.defenseAction(gameState) \
            if self._defense \
            else self.offenseAction(gameState)

    def defenseAction(self, gameState):
        """
        Choose a defensive action
        """
        index = self.index
        data = gameState.data
        agentStates = data.agentStates
        agentState = agentStates[index]
        pos = agentState.configuration.pos
        distancer = self.distancer

        # compute the expected distances to each opponent according to the
        # current distribution
        expected = [
            (k, max(dist.items(), key=itemgetter(1))[0])
            for k, dist in self._distribution.items()
        ]
        closest = min(
            (
                # try to reach the closest agent and select the one carrying
                # more food if ties
                (p, distancer.getDistance(pos, p), agentStates[k].numCarrying)
                for k, p in expected
            ),
            key=itemgetter(1, 2)
        )[0]
        # generate state for each successor
        a = [
            (a, gameState.generateSuccessor(index, a))
            for a in gameState.getLegalActions(index)
        ]
        # filter out the state where the agent has crossed the bound
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
        print("----", pos)
        for i in self._oppo:
            conf = agentStates[i].configuration
            initPos = data.layout.agentPositions[i][1]
            if conf:
                print("----", i, conf.pos)
                if conf.pos == pos:
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


def _MCTS(index, gameState, judge, depth=5, count=100):
    actions = gameState.getLegalActions(index)
    scores = [0] * len(actions)
    for i, action in enumerate(actions):
        for c in xrange(count):
            ind, d = index, depth
            successor = gameState.generateSuccessor(ind, action)
            if not successor.isOver():
                while d > 1:
                    ind = (ind + 1) % 4
                    successor = successor.generateSuccessor(
                        ind, random.choice(successor.getLegalActions(ind))
                    )
                    if successor.isOver():
                        break
                    d -= 1
            scores[i] += judge(gameState, successor)
    print(scores)
    ms = max(scores)
    return random.choice([a for a, s in zip(actions, scores) if s == ms])


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

        # record each instance created
        self._instances[index // 2] = self
        self._teammate = (index // 2 + 1) % 2

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

    def _judge(self, gameState, successor):
        index = self.index
        gdata, sdata = gameState.data, successor.data
        sp = sdata.score - gdata.score
        nc = sdata.agentStates[index].numCarrying
        ncp = nc - gdata.agentStates[index].numCarrying
        if sp == 0:
            if ncp < 0:
                return -1
            return nc * 0.5 + 1
        ncp += sp
        s = (sp + ncp / 2) * self._height * self._width
        agentStates = sdata.agentStates
        distancer = self.distancer
        pos = agentStates[index].configuration.pos
        s += sum(
            distancer.getDistance(pos, agentStates[i].configuration.pos)
            for i in (gameState.blueTeam if self.red else gameState.redTeam)
        )
        return s

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

        if self._prevCarry > agentState.numCarrying:
            self._computeRoute(gameState)
        self._prevCarry = agentState.numCarrying

        distancer = self.distancer
        pos = agentState.configuration.pos
        dist = [
            distancer.getDistance(agentStates[i].configuration.pos, pos)
            for i in (gameState.blueTeam if red else gameState.redTeam)
        ]
        if any(d < 5 for d in dist) and agentState.numCarrying > 0:
            return _MCTS(index, gameState, self._judge, 12)

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
