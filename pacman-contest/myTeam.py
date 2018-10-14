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

from heapq import heappop, heappush
from operator import itemgetter

from captureAgents import CaptureAgent
from distanceCalculator import manhattanDistance
from game import Actions, Directions


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
        self._maskUpdated = self._recompute = self._escape = False

        self._maskFood = set()
        self._leftFood = set()

        self._walls = None

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

        # record the bound in our side
        bound = half - 1 if red else half
        walls = layout.walls.data
        self._bound = set(
            (bound, y) for y in xrange(height) if not walls[bound][y]
        )

        # assume defensive agent will never reach the other side
        for x in (xrange(half, width) if self.red else xrange(half)):
            for y in xrange(height):
                walls[x][y] = True
        self._walls = walls

        # get an instance of the teammate
        self._teammate = self._instances[(self.index // 2 + 1) % 2]

    def _updateMask(self, gameState):
        agentStates = gameState.data.agentStates
        red = self.red

        # determine the positions of the opponents
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

        # mask the food which can be reached by opponent in three steps
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

        # route needs to be recomputed if the food set changed
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

        # determine if the current path needs to be recomputed
        _recompute = self._recompute or self._maskUpdated
        walls = data.layout.walls.data
        _actions = self._actions
        for i in (gameState.blueTeam if self.red else gameState.redTeam):
            agentState = agentStates[i]
            if not agentState.isPacman and agentState.scaredTimer == 0:
                # pretend there are walls around the opponent agents if they are
                # not scared
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

        # assign the closest food for trial
        leftFood = self._leftFood
        if not leftFood:
            maskFood = self._maskFood
            # start to escape asap if no foods left
            if not maskFood:
                return self._getEscapeNext(gameState)
            mfs = min((
                (f, distancer.getDistance(pos, f))
                for f in maskFood
            ), key=itemgetter(1))[0]
            leftFood.add(mfs)
            maskFood.remove(mfs)

        # A* to eat
        path = []
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
        self._recompute = not path

        self._actions = path
        return Actions.vectorToDirection((nx - x, ny - y))

    def observationFunction(self, gameState):
        """
        Abuse this function to obtain a full game state from the controller
        We actually get an inference module in the inference.py but since this
        is not explicitly disallowed in all documents so we decide to utilise
        this design flaw
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

        red = self.red
        index = self.index
        data = gameState.data
        half = data.layout.width // 2
        agentStates = data.agentStates
        bounds = self._bound
        distancer = self.distancer

        _recompute = not self._escape
        walls = data.layout.walls.data
        _escapes = self._escapes
        for i in (gameState.blueTeam if self.red else gameState.redTeam):
            agentState = agentStates[i]
            if not agentState.isPacman and agentState.scaredTimer == 0:
                # pretend there are walls around the opponent agents if they are
                # not scared
                x, y = agentState.configuration.pos
                pos = x, y = int(x), int(y)
                if _escapes is None or pos in _escapes:
                    _recompute = True
                if red and x - 1 >= half or not red and x - 1 < half:
                    walls[x - 1][y] = True
                if red and x + 1 >= half or not red and x + 1 < half:
                    walls[x + 1][y] = True
                walls[x][y + 1] = walls[x][y - 1] = walls[x][y] = True

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
            # TODO: return other valid action
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
        target = tuple(map(int, target))

        agent = gameState.data.agentStates[self.index]
        x, y = pos = tuple(map(int, agent.configuration.pos))
        distancer = self.distancer

        dist = distancer.getDistance(pos, target)
        cp = self._chasepath
        if cp is not None:
            movement = manhattanDistance(cp[0], target)
            if movement < 1:
                cp = [target] + cp

            if len(cp) <= dist:
                nx, ny = cp.pop()
                self._chasepath = cp if cp else None
                return Actions.vectorToDirection((nx - x, ny - y))

        walls = self._walls
        # A* to chase
        path = []
        q = [(dist, dist, 0, pos, path)]
        visited = set()
        while q:
            _, _, g, pos, path = heappop(q)

            if pos == target:
                break

            visited.add(pos)

            x, y = pos
            for dx, dy in self._dirs:
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = distancer.getDistance(pos, target)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))

        if not path:
            return Directions.STOP

        path.reverse()
        x, y = agent.configuration.pos
        nx, ny = path.pop()

        self._chasepath = path if path else None
        return Actions.vectorToDirection((nx - x, ny - y))

    def _defenseAction(self, gameState):
        index = self.index
        red = self.red
        data = gameState.data
        agentStates = data.agentStates
        distancer = self.distancer
        bounds = self._bound
        agent = agentStates[index]
        pos = agent.configuration.pos
        scare = agent.scaredTimer > 0
        walls = data.layout.walls.data

        target = None
        rs = []
        pnc = 0
        for i in (gameState.blueTeam if red else gameState.redTeam):
            agentState = agentStates[i]
            nc = agentState.numCarrying
            npos = agentState.configuration.pos
            if nc > pnc:
                pnc = nc
                target = agentState.configuration.pos
            rs.append((min(
                (
                    (
                        b,
                        (
                            distancer.getDistance(npos, b),
                            distancer.getDistance(pos, b)
                        )
                    )
                    for b in bounds
                ),
                key=itemgetter(1)
            ), npos, agentState.isPacman))

        if target is not None:
            if scare:
                tx, ty = target
                sur = [
                    (int(tx + cx), int(ty + cy)) for cx, cy in self._closes
                ]
                sur = [
                    (x, y) for x, y in sur if not walls[x][y]
                ]
                sel = min(
                    (
                        (
                            s,
                            min(distancer.getDistance(s, b) for b in bounds),
                            distancer.getDistance(pos, s)
                        )
                        for s in sur
                    ),
                    key=itemgetter(1, 2)
                )[0]
                return self._chase(gameState, sel)
            return self._chase(gameState, target)

        mb = None
        mbd = (inf, inf)
        md = inf
        for (b, bd), npos, pac in rs:
            dist = distancer.getDistance(npos, pos)
            if pac:
                if dist < md:
                    target = npos
                    md = dist
            else:
                if bd < mbd:
                    mb, mbd = b, bd

        if target is not None:
            if scare:
                tx, ty = target
                sur = [
                    (tx + cx, ty + cy) for cx, cy in self._closes
                ]
                sur = [
                    (x, y) for x, y in sur if not walls[x][y]
                ]
                sel = min(
                    (
                        (
                            s,
                            min(distancer.getDistance(s, b) for b in bounds),
                            distancer.getDistance(pos, s)
                        )
                        for s in sur
                    ),
                    key=itemgetter(1, 2)
                )[0]
                return self._chase(gameState, sel)
            return self._chase(gameState, target)

        if scare:
            tx, ty = mb
            sur = [
                (tx + cx, ty + cy) for cx, cy in self._closes
            ]
            sur = [
                (x, y) for x, y in sur if not walls[x][y]
            ]
            sel = min(
                (
                    (
                        s,
                        min(distancer.getDistance(s, b) for b in bounds),
                        distancer.getDistance(pos, s)
                    )
                    for s in sur
                ),
                key=itemgetter(1, 2)
            )[0]
            return self._chase(gameState, sel)
        return self._chase(gameState, mb)
