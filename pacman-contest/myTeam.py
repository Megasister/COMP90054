# myTeam.py
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
from distanceCalculator import manhattanDistance
from game import Directions, Actions


################################################################################
# Constants
################################################################################
inf = float("inf")


################################################################################
# Team creation
################################################################################
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
# Mixed Agents
################################################################################
class AbuseAStarAgent(CaptureAgent, object):
    """
    An agent perform an offensive/defensive action depends on different
    predefined circumstances, mainly use A* as the strategy to indicate the
    next step
    """
    __slots__ = ()

    _instances = [None, None]

    def __init__(self, index, red, defense, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        object.__init__(self)

        self.red = red
        self._defense = defense
        self._height = self._width = self._half = self._bound = \
            self._actions = self._escapes = None
        self._prevCarry = 0
        self._recompute = False
        self._escape = False

        self._mask_food = set()
        self._left_food = set()

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
        walls = layout.walls.data
        self._bound = set(
            (bound, y) for y in xrange(height) if not walls[bound][y]
        )

        agentStates = data.agentStates
        poss = [
            agentStates[i].configuration.pos
            for i in (gameState.blueTeam if red else gameState.redTeam)
        ]
        distancer = self.distancer
        food = layout.food.data
        for x in (xrange(half, width) if red else xrange(half)):
            for y in xrange(height):
                if food[x][y]:
                    pos = x, y
                    if any(distancer.getDistance(pos, p) < 6 for p in poss):
                        self._mask_food.add(pos)
                    else:
                        self._left_food.add(pos)

        # only offensive agent needs to compute the route
        if not self._defense:
            self._computeFoodRoute(gameState)

    def _computeFoodRoute(self, gameState):
        if not self._updateMask and not self._recomputeFR:
            return self

        data = gameState.data
        bounds = self._bound
        distancer = self.distancer

        # Dijkstra (or variant of Uniform Cost Search) implementation
        pos = data.agentStates[self.index].configuration.pos
        path = []
        q = [(0, pos, path, self._left_food)]
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

        return self

    def _monitorMask(self, gameState):
        agentStates = gameState.data.agentStates
        poss = [
            agentStates[i].configuration.pos
            for i in (gameState.blueTeam if red else gameState.redTeam)
        ]

        distancer = self.distancer
        added = set()
        for f in self._mask_food:
            if all(distancer.getDistance(f, p) > 5 for p in poss):
                added.add(f)

        deleted = set()
        for f in self._left_food:
            if any(distancer.getDistance(f, p) < 6 for p in poss):
                deleted.add(f)

        if added:
            self._mask_food -= added
            self._left_food |= added
            self._updateMask = True

        if deleted:
            self._mask_food |= deleted
            self._left_food -= deleted
            self._updateMask = True

        return self

    def _getFoodNext(self, gameState):
        index = self.index
        agentState = gameState.data.agentStates[index]
        distancer = self.distancer
        actions = self._actions
        if agentState.configuration.pos == actions[-1]:
            actions.pop()
        cdest = actions[-1]
        successors = [
            (a, gameState.generateSuccessor(index, a))
            for a in gameState.getLegalActions(index)
        ]
        return min(
            (
                (
                    a,
                    distancer.getDistance(
                        succ.data.agentStates[index].configuration.pos, cdest
                    ),
                    len(succ.getLegalActions(index))
                )
                for a, succ in successors
            ),
            key=itemgetter(1, 2)
        )[0]

    def observationFunction(self, gameState):
        """
        Abuse this function to obtain a full game state from the controller
        """
        return gameState

    def chooseAction(self, gameState):
        """
        Choose an action based on the current status of the agent
        """

        return self._defenseAction(gameState) \
            if self._defense \
            else self._offenseAction(gameState)

    def _getEscapeNext(self, gameState):
        index = self.index
        data = gameState.data
        agentStates = data.agentStates
        bounds = self._bound
        distancer = self.distancer

        _recompute = self._recomputeER
        if not _recompute:
            walls = data.layout.walls.data
            for i in (gameState.blueTeam if self.red else gameState.redTeam):
                agentState = agentStates[i]
                if not agentState.isPacman:
                    x, y = agentState.configuration.pos
                    pos = x, y = int(x), int(y)
                    if pos in self._escapes:
                        _recompute = True
                        break
                    walls[x + 1][y] = walls[x][y + 1] = walls[x - 1][y] = \
                        walls[x][y - 1] = walls[x][y] = True

        agent = agentStates[index]
        x, y = pos = tuple(map(int, agent.configuration.pos))
        if not _recompute:
            nx, ny = self._escapes.pop()
            return Actions.vectorToDirection((nx - x, ny - y))

        # A* to escape
        path = []
        h = min(distancer.getDistance(pos, b) for b in bounds)
        q = [(h, h, 0, pos, path)]
        escaped = False
        visited = set()
        while q:
            _, _, g, pos, path = heappop(q)

            if pos in bounds:
                escaped = True
                break

            visited.add(pos)

            x, y = pos
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = min(distancer.getDistance(npos, b) for b in bounds)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))

        if not escaped:
            self._defense = self._instances[
                (index // 2 + 1) % 2
            ]._notifyFailEscape()
            return None

        path.reverse()
        x, y = agent.configuration.pos
        nx, ny = path.pop()
        if not path:
            self._escape = False

        return Actions.vectorToDirection((nx - x, ny - y))

    def _offenseAction(self, gameState):
        if self._escape:
            return self._getEscapeNext(gameState)

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
            nc = self._prevCarry = agentState.numCarrying
            if nc > 0:
                self._escape = True
                return self._getEscapeNext(gameState)

        return self._monitorMask(gameState)._computeFoodRoute(
            gameState
        )._getFoodNext(gameState)

    def _target(self, gameState, target):
        index = self.index
        agentStates = gameState.data.agentStates
        agent = agentStates[index]
        food = agentStates[target]

        penalty = 2 if agent.scaredTimer > 0 else 0
        fpos = food.configuration.pos

        distancer = self.distancer
        successors = [
            (a, gameState.generateSuccessor(index, a))
            for a in gameState.getLegalActions(index)
        ]
        successors = [
            (a, successor)
            for a, successor in successors
            if not successor.data.agentStates[index].isPacman
        ]
        dist = [
            (
                a, distancer.getDistance(
                    succ.data.agentStates[index].configuration.pos, fpos
                )
            ) for a, succ in successors
        ]
        dist = [(d, a) for a, d in dist if d >= penalty]

        minval = min(dist)[0]
        return random.choice([a for d, a in dist if d == minval])

    def _defenseAction(self, gameState):
        index = self.index
        red = self.red
        data = gameState.data
        agentStates = data.agentStates
        agentState = agentStates[index]
        pos = agentState.configuration.pos
        distancer = self.distancer

        nt = 0
        t = None
        pnc = 0
        target = []
        for i in (gameState.blueTeam if red else gameState.redTeam):
            agentState = agentStates[i]
            nc = agentState.numCarrying
            if nc > 0:
                nt += 1
                if nc > pnc:
                    pnc = nc
                    t = i
            target.append((i, agentState))

        if nt > 0:
            return self._target(gameState, t)

        dist = [
            (
                i,
                -a.isPacman,
                distancer.getDistance(
                    pos, a.configuration.pos
                )
            )
            for i, a in target
        ]

        return self._target(gameState, min(dist, key=itemgetter(1, 2))[0])
