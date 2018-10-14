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
from baselineTeam import DefensiveReflexAgent
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
    first='AbuseAStarAgent',
    second='AbuseAStarAgent',
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

    _instances = [None, None]
    
    _dirs = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    _closes = {
        (2, 0), (0, 2), (-2, 0), (0, -2), (1, 1), (-1, -1), (1, -1), (-1, 1)
    }

    def __init__(self, index, red, defense, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        object.__init__(self)

        self.red = red
        self._defense = defense
        self._height = self._width = self._half = self._bound = \
            self._actions = self._escapes = self._teammate = \
            self._prevPos = self._chasepath = None
        self._prevCarry = 0
        self._chasing = self._maskUpdated = self._recompute = \
            self._escape = False

        self._maskFood = set()
        self._leftFood = set()

        # record each instance created
        self._instances[index // 2] = self

    def registerInitialState(self, gameState):
        """
        Initialise the agent and compute an initial route
        """
        CaptureAgent.registerInitialState(self, gameState)

        data = gameState.data
        layout = data.layout
        height = layout.height
        width = layout.width
        half = width // 2
        red = self.red
        bound = half - 1 if red else half
        walls = layout.walls.data
        self._bound = set(
            (bound, y) for y in xrange(height) if not walls[bound][y]
        )

        self._teammate = self._instances[(self.index // 2 + 1) % 2]

    def _updateMask(self, gameState):
        agentStates = gameState.data.agentStates
        red = self.red
        poss = [
            agentStates[i].configuration.pos
            for i in (gameState.blueTeam if red else gameState.redTeam)
        ]

        distancer = self.distancer
        data = gameState.data
        layout = data.layout
        width = layout.width
        half = width // 2
        height = layout.height
        food = data.food
        _maskFood = set()
        _leftFood = set()
        for x in (xrange(half, width) if red else xrange(half)):
            for y in xrange(height):
                if food[x][y]:
                    pos = x, y
                    if any(distancer.getDistance(pos, p) < 6 for p in poss):
                        _maskFood.add(pos)
                    else:
                        _leftFood.add(pos)

        if _maskFood == self._maskFood:
            self._maskUpdated = True
        else:
            self._maskUpdated = False

        self._maskFood = _maskFood
        self._leftFood = _leftFood

        return self

    def _getFoodNext(self, gameState):
        self._escape = False

        index = self.index
        red = self.red
        data = gameState.data
        half = data.layout.width // 2
        agentStates = data.agentStates
        distancer = self.distancer

        _recompute = self._recompute or self._maskUpdated
        walls = data.layout.walls.data
        _actions = self._actions
        for i in (gameState.blueTeam if self.red else gameState.redTeam):
            agentState = agentStates[i]
            if not agentState.isPacman and agentState.scaredTimer == 0:
                x, y = agentState.configuration.pos
                pos = x, y = int(x), int(y)
                if _actions is None or pos in _actions:
                    _recompute = True
                if red and x - 1 >= half or not red and x - 1 < half:
                    walls[x - 1][y] = True
                if red and x + 1 >= half or not red and x + 1 < half:
                    walls[x + 1][y] = True
                walls[x][y + 1] = walls[x][y - 1] = walls[x][y] = True

        agent = agentStates[index]
        x, y = pos = tuple(map(int, agent.configuration.pos))
        if not _recompute:
            nx, ny = _actions.pop()
            if not _actions:
                self._recompute = True
            return Actions.vectorToDirection((nx - x, ny - y))

        # A* to eat
        path = []
        leftFood = self._leftFood
        h = min(distancer.getDistance(pos, f) for f in leftFood)
        q = [(h, h, 0, pos, path)]
        visited = set()
        while q:
            _, _, g, pos, path = heappop(q)

            if pos in leftFood:
                break

            visited.add(pos)

            x, y = pos
            for dx, dy in self._dirs:
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = min(distancer.getDistance(npos, f) for f in leftFood)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))

        path.reverse()
        x, y = agent.configuration.pos
        nx, ny = path.pop()
        if not path:
            self._recompute = True
        else:
            self._recompute = False

        self._actions = path
        return Actions.vectorToDirection((nx - x, ny - y))

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
        self._recompute = True

        index = self.index
        data = gameState.data
        agentStates = data.agentStates
        bounds = self._bound
        distancer = self.distancer

        _recompute = not self._escape
        walls = data.layout.walls.data
        _escapes = self._escapes
        for i in (gameState.blueTeam if self.red else gameState.redTeam):
            agentState = agentStates[i]
            if not agentState.isPacman:
                x, y = agentState.configuration.pos
                pos = x, y = int(x), int(y)
                if _escapes is None or pos in _escapes:
                    _recompute = True
                walls[x + 1][y] = walls[x][y + 1] = walls[x - 1][y] = \
                    walls[x][y - 1] = walls[x][y] = True

        agent = agentStates[index]
        x, y = pos = tuple(map(int, agent.configuration.pos))
        if not _recompute:
            nx, ny = _escapes.pop()
            if not _escapes:
                self._escape = False
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
            for dx, dy in self._dirs:
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = min(distancer.getDistance(npos, b) for b in bounds)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))

        if not escaped:
            # self._teammate.notifyEscFail()
            self._recompute = True
            self._escape = False
            return Directions.STOP

        path.reverse()
        x, y = agent.configuration.pos
        nx, ny = path.pop()
        if not path:
            self._escape = False
        else:
            self._escape = True

        self._escapes = path
        return Actions.vectorToDirection((nx - x, ny - y))

    def _offenseAction(self, gameState):
        index = self.index
        red = self.red
        agentStates = gameState.data.agentStates
        agentState = agentStates[index]

        distancer = self.distancer
        pos = agentState.configuration.pos

        _prevPos = self._prevPos
        if _prevPos is not None:
            if manhattanDistance(pos, _prevPos) > 1:
                self._escape = False
                self._recompute = True
                # self._teammate._notifyReborn()

        self._prevPos = pos

        if self._escape:
            return self._getEscapeNext(gameState)

        states = [
            agentStates[i]
            for i in (gameState.blueTeam if red else gameState.redTeam)
        ]
        if any(
            not s.isPacman and s.scaredTimer == 0 and distancer.getDistance(
                s.configuration.pos, pos
            ) < 4
            for s in states
        ):
            self._recompute = True
            nc = self._prevCarry = agentState.numCarrying
            if nc > 0:
                return self._getEscapeNext(gameState)

        return self._updateMask(gameState)._getFoodNext(gameState)

    def _chase(self, gameState, target):
        data = gameState.data
        layout = data.layout
        agentStates = data.agentStates
        agent = agentStates[self.index]
        x, y = pos = tuple(map(int, agent.configuration.pos))
        _target = agentStates[target]
        tpos = _target.configuration.pos
        distancer = self.distancer

        dist = distancer.getDistance(pos, tpos)
        cp = self._chasepath
        if self._chasing:
            movement = manhattanDistance(cp[0], tpos)
            if movement < 1:
                cp = [tpos] + cp
            elif movement > 1:
                self._chasing = False

        if self._chasing:
            if len(cp) <= dist:
                nx, ny = cp.pop()
                self._chasepath = cp
                return Actions.vectorToDirection((nx - x, ny - y))

        walls = layout.walls.data
        height, width = layout.height, layout.width
        half = width // 2
        for x in (xrange(half, width) if self.red else xrange(half)):
            for y in xrange(height):
                walls[x][y] = True

        # A* to chase
        path = []
        q = [(dist, dist, 0, pos, path)]
        visited = set()
        while q:
            _, _, g, pos, path = heappop(q)

            if pos == tpos:
                break

            visited.add(pos)

            x, y = pos
            for dx, dy in self._dirs:
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = distancer.getDistance(pos, tpos)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))

        path.reverse()
        x, y = agent.configuration.pos
        nx, ny = path.pop()
        if not path:
            self._chasing = False
        else:
            self._chasing = True

        self._chasepath = path
        return Actions.vectorToDirection((nx - x, ny - y))

    def _chaseBound(self, gameState, target):
        pass

    def _scareChase(self, gameState, target):
        data = gameState.data
        layout = data.layout
        agentStates = data.agentStates
        agent = agentStates[self.index]
        x, y = pos = agent.configuration.pos
        _target = agentStates[target]
        tx, ty = tpos = _target.configuration.pos
        distancer = self.distancer

        walls = layout.walls.data
        height, width = layout.height, layout.width
        half = width // 2
        for x in (xrange(half, width) if self.red else xrange(half)):
            for y in xrange(height):
                walls[x][y] = True

        dist = distancer.getDistance(pos, tpos)
        dests = set((tx + cx, ty + cy) for cx, cy in self._closes)

        # A* to chase
        path = []
        q = [(dist, dist, 0, pos, path)]
        visited = set()
        while q:
            _, _, g, pos, path = heappop(q)

            if pos in dests:
                break

            visited.add(pos)

            x, y = pos
            for dx, dy in self._dirs:
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = distancer.getDistance(pos, tpos)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))

        x, y = agent.configuration.pos
        nx, ny = path[0]
        return Actions.vectorToDirection((nx - x, ny - y))

    def _defenseAction(self, gameState):
        index = self.index
        red = self.red
        data = gameState.data
        agentStates = data.agentStates
        agent = agentStates[index]
        pos = agent.configuration.pos
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

        if nt == 0:
            t = min((
                (
                    i,
                    -a.isPacman,
                    distancer.getDistance(pos, a.configuration.pos)
                )
                for i, a in target
            ), key=itemgetter(1, 2))[0]

        if agent.scaredTimer > 0:
            return self._scareChase(gameState, t)
        return self._chase(gameState, t)
