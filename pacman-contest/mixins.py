from __future__ import division, print_function
import random
from abc import ABCMeta, abstractmethod
from operator import add, itemgetter
from captureAgents import CaptureAgent
from distanceCalculator import manhattanDistance
from game import Directions


##########
# Agents #
##########

################################################################################
# Mixins
################################################################################
# force new style Python 2 class by multiple inheritance
class EvalBaseMixin(object):
    """
    This is a mixin generalising all agents make decision based on evaluation as
    a dot product of features
    """
    def _eval(self, gameState, action):
        return sum(
            f * v
            for f, v in zip(
                self._getFeatures(gameState, action), self._weights
            )
        )


class GreedyDelegate(object):
    """
    This is a mixin defining a greedy strategy to choose an action based on
    evaluation of the next state only
    """
    def chooseAction(self, gameState):
        """
        Greedily choose the action leads to best successor state from current
        game state
        """
        # greedy strategy
        combs = [
            (a, self._eval(gameState, a))
            for a in gameState.getLegalActions(self.index)
        ]
        print(combs)
        maxVal = max(map(itemgetter(1), combs))
        # randomly break ties if exist
        return random.choice([a for a, v in combs if v == maxVal])


# helper function to perform element-wise addition on two lists
def _sum_list(x, y):
    return map(add, x, y)


class OffensiveMixin(object):
    """
    Mixin for offensive features generation
    """
    def __init__(self):
        self._weights = None

    def _initialiseWeights(self):
        maxDist = self._maxDist
        self._weights = [
            maxDist * 2,
            -100,
            100,
            -5,
            -2,
            maxDist,
            1,
            1,
            -10, -8, -6, -4, -2
        ]

    def _getFeatures(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        self._updateDistribution(successor)

        features = [0] * 13

        data = successor.data

        ########################################################################
        # general features
        ########################################################################
        # score
        features[0] = data.score
        # determine if the action is a STOP
        features[1] = action == Directions.STOP

        agentState = data.agentStates[self.index]
        p = agentState.isPacman
        # determine if currently is a Pacman
        features[2] = p

        ########################################################################
        # offensive features
        ########################################################################
        height = self._height
        width = self._width
        half = width // 2
        foods = data.food.data
        currPos = agentState.configuration.pos
        distancer = self.distancer
        maxDist = self._maxDist
        red = self.red

        # the closest food
        m = maxDist
        for x in (xrange(half, width) if red else xrange(half)):
            for y in xrange(height):
                if foods[x][y]:
                    m = min(m, distancer.getDistance(currPos, (x, y)))
        features[3] = m

        # the closest capsule
        m = maxDist
        for x, y in data.capsules:
            if not red and x < half or red and x >= half:
                m = min(m, distancer.getDistance(currPos, (x, y)))
        features[4] = m

        # the number of food carrying
        features[5] = agentState.numCarrying

        # the closest distance to the boundary to return the carry food
        walls = data.layout.walls.data
        b = half - 1 if red else half
        features[6] = min(
            distancer.getDistance(currPos, (b, y))
            for y in xrange(height) if not walls[b][y]
        )

        if p:
            pass
        else:
            pass

        return features


class InferenceMixin(object):
    """
    Mixin for probability inference of positions of opponent agents
    """
    __metaclass__ = ABCMeta

    _DIRECTION = (
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0)
    )

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

    def __init__(self):
        self._distribution = self._validPos = None

    def _initialise(self, initState):
        # initialise all valid positions with the given wall configuration
        data = initState.data
        wall = data.layout.walls.data
        self._validPos = [
            (x, y)
            for x in xrange(self._width)
            for y in xrange(self._height)
            if not wall[x][y]
        ]

        # the initial position is visible
        agentStates = data.agentStates
        oppo = initState.blueTeam if self.red else initState.redTeam
        self._distribution = {
            i: {tuple(map(int, agentStates[i].configuration.pos)): 1.0}
            for i in oppo
        }

    def _updateDistribution(self, gameState):
        agentDistances = gameState.agentDistances
        data = gameState.data
        layout = data.layout
        wall = layout.walls.data
        agentStates = data.agentStates
        vp = self._validPos
        dist = {}
        pos = agentStates[self.index].configuration.pos

        for agent, adist in self._distribution.items():
            agentState = agentStates[agent]
            conf = agentState.configuration
            # if the agent is visible
            if conf is not None:
                dist[agent] = {tuple(map(int, conf.pos)): 1.0}
                continue

            ndist = agentDistances[agent]
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


class AbuseDelegate(object):
    """
    This delegation mixin implement a very simple observation function, which
    may abuse the actual game rule and enable a god-view observation
    """
    __metaclass__ = ABCMeta
    __slots__ = ()

    def observationFunction(self, gameState):
        return gameState


################################################################################
# end
################################################################################
