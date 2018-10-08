# testTeam.py
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
from abc import ABCMeta, abstractmethod
from baselineTeam import DefensiveReflexAgent
from captureAgents import CaptureAgent
from distanceCalculator import manhattanDistance
from game import Directions
from operator import itemgetter, add
import random

infinity = float("inf")


#################
# Team creation #
#################
def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first='OffensiveGreedyAgent',
    second='DefensiveReflexAgent'
):
    return [eval(first)(firstIndex, isRed), eval(second)(secondIndex, isRed)]


##########
# Agents #
##########

################################################################################
# Abstract Classes / Mixins
################################################################################
# force new style Python 2 class by multiple inheritance
class GreedyAgent(CaptureAgent, object):
    """
    This is an abstract class generalising all agents make decision based on
    evaluation as a dot product of features and weights
    """
    __metaclass__ = ABCMeta
    __slots__ = "_half", "_height", "_maxDist", "_weights", "_width"

    def __init__(self, index, red, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        object.__init__(self)

        self._half = self._height = self._maxDist = self._weights = \
            self._width = None

        self.red = red

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        data = gameState.data
        layout = data.layout
        self._height = layout.height
        width = self._width = layout.width
        self._half = width // 2
        self._maxDist = self._height * width

    @abstractmethod
    def evaluate(self, gameState, action):
        """
        Evaluate and compute a score with a given action and game state by
        generating features
        """
        pass

    def chooseAction(self, gameState):
        """
        Greedily choose the action leads to best successor state from current
        game state
        """
        # greedy strategy
        combs = [
            (a, self.evaluate(gameState, a))
            for a in gameState.getLegalActions(self.index)
        ]
        print(combs)
        maxVal = max(map(itemgetter(1), combs))
        # randomly break ties if exist
        return random.choice([a for a, v in combs if v == maxVal])


# helper function to perform element-wise addition on two lists
def _sum_list(x, y):
    return map(add, x, y)


class InferenceMixin(object):
    """
    Mixin for probability inference of positions of opponent agents
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._validPos = None

    def _initialiseValidPos(self, wall):
        # initialise all valid positions with the given wall configuration
        self._validPos = [
            (x, y)
            for x in xrange(self._width)
            for y in xrange(self._height)
            if not wall[x][y]
        ]

    def _observerAgent(self, pos, ndist, agenState):
        # if the agent is visible return the position directly
        conf = agenState.configuration
        if conf is not None:
            return {conf.pos: 1}

        # obtain all possible positions with the given distance
        vp = self._validPos
        ps = [p for p in vp if manhattanDistance(pos, p) == ndist]

        # test each point if it is in the noisy distance to the points above
        freq = {
            p: [manhattanDistance(p, ip) < 7 for ip in ps]
            for p in vp
        }

        # count the valid points to each possible position then obtain each of
        # the probabilities
        sl = len(ps)
        s = [1 / v / sl for v in reduce(_sum_list, freq.values())]

        # sum up the probabilities
        return {
            k: sum(iv * si for iv, si in zip(v, s))
            for k, v in freq.items()
            # ignore position with probability of 0
            if any(v)
        }

    def _observeState(self, gameState):
        dists = gameState.agentDistances
        agentStates = gameState.data.agentStates
        pos = agentStates[self.index].configuration.pos
        return {
            i: self._observerAgent(pos, dists[i], agentStates[i])
            for i in (gameState.blueTeam if self.red else gameState.redTeam)
        }


class ExpectiMin(object):
    """
    A mixin implement the Expecti Min adversarial search and assuming the class
    have a self.evaluate method
    """
    __metaclass__ = ABCMeta
    __slots__ = ()

    def expectMin(self):
        pass

    def evalMax(self):
        pass


class WeightTrainableAgent(object):
    """
    This is an abstract class generalising all agents make decision based on
    evaluation and it is trainable (specifically w.r.t weights)
    """
    __metaclass__ = ABCMeta
    __slots__ = ()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass


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
# Concrete Agents
################################################################################
class OffensiveGreedyAgent(GreedyAgent, InferenceMixin):
    """
    An agent greedily choose the maximum score of next step only with
    offensive strategy
    """
    __slots__ = "_teamTotalFood"

    def __init__(self, index, red, timeForComputing=.1):
        GreedyAgent.__init__(self, index, timeForComputing)
        InferenceMixin.__init__(self)

        self._teamTotalFood = 0

    def _countFood(self, foods, half, width):
        red = self.red
        self._teamTotalFood = sum(
            f for x in (xrange(half, width) if red else xrange(half))
            for f in foods[x]
        )

    def registerInitialState(self, gameState):
        """
        Initialise the agent with the given state and initialise a list of
        weights
        """
        GreedyAgent.registerInitialState(self, gameState)

        data = gameState.data
        self._initialiseValidPos(data.layout.walls.data)
        self._countFood(data.food.data, self._half, self._width)
        print(self._validPos)

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

    def evaluate(self, gameState, action):
        """
        Generates a vector of features then return the dot product of it and the
        weight vector
        """
        print(self._observeState(gameState))
        features = [0] * 13

        index = self.index
        successor = gameState.generateSuccessor(index, action)

        data = successor.data

        ########################################################################
        # general features
        ########################################################################
        # score
        features[0] = data.score
        # determine if the action is a STOP
        features[1] = action == Directions.STOP

        agentState = data.agentStates[index]
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
        m = maxDist
        for x in (xrange(half, width) if red else xrange(half)):
            for y in xrange(height):
                if foods[x][y]:
                    m = min(m, distancer.getDistance(currPos, (x, y)))
        # the most distant food
        features[3] = m

        m = maxDist
        for x, y in data.capsules:
            if not red and x < half or red and x >= half:
                m = min(m, distancer.getDistance(currPos, (x, y)))
        # the most distant capsule
        features[4] = m
        features[5] = agentState.numCarrying

        # the closest distance to the boundary to return the carry food
        walls = data.layout.walls.data
        b = half - 1 if red else half
        features[6] = min(
            distancer.getDistance(currPos, (b, y))
            for y in xrange(height) if not walls[b][y]
        )

        if p:
            for i in (gameState.blueTeam if red else gameState.redTeam):
                s = data.agentStates[i]
                conf = s.configuration
                pos = conf.pos if conf else None
                st = s.scaredTimer
                print(st)
                if st == 0:
                    if pos is None:
                        # could consider the noisy distance distribution
                        pass
                    else:
                        dist = distancer.getDistance(pos, currPos)
                        print("    " + str(dist))
                        # if the opponent agent is not currently a ghost
                        # if the opponent is not currently in the same side
                        if s.isPacman:
                            # determine the closest distance to the boundary
                            ob = half if red else half - 1
                            odb = min(
                                distancer.getDistance(pos, (ob, y))
                                for y in xrange(height) if not walls[ob][y]
                            )
                            if odb < 2:
                                features[7 + dist] += 1
                        else:
                            if dist < 6:
                                features[7 + dist] += 1
                else:
                    # opponent will not affect current state
                    features[7] += 1
        else:
            # could consider defensive operation as well
            pass

        ########################################################################
        # dot product
        ########################################################################
        print(action, len(features), features, len(self._weights), self._weights)
        return sum(f * w for f, w in zip(features, self._weights))


class DefensiveGreedyAgent(GreedyAgent, InferenceMixin):
    """
    An agent greedily choose the maximum score of next step only with
    defensive strategy
    """
    __slots__ = "_weights"

    def __init__(self, index, red, timeForComputing=.1):
        GreedyAgent.__init__(self, index, red, timeForComputing)
        InferenceMixin.__init__(self)

        self._weights = None

    def registerInitialState(self, gameState):
        """
        Initialise the agent with the given state and initialise a list of
        weights
        """
        GreedyAgent.registerInitialState(self, gameState)

        self._initialiseValidPos(gameState.data.walls.data)

        self._weights = [
        ]

    def evaluate(self, gameState, action):
        """
        The list of features are
        [0]:    current score
        (offensive)
        [1]:    1 if the agent is currently a pacman, 0 otherwise
        [2]:    negation of [1]
        (general)
        [3]:    1 if action is stop else 0
        """
        features = [0] * 4

        successor = gameState.generateSuccessor(self.index, action)

        data = successor.data
        agentState = successor.getAgentState(self.index)

        features[0] = data.score

        p = agentState.isPacman
        features[1] = p
        features[2] = not p

        ########################################################################
        # defensive features
        ########################################################################

        ########################################################################
        # general features
        ########################################################################
        features[3] = action == Directions.STOP
        return sum(f * w for f, w in zip(features, self._weights))


################################################################################
# end
################################################################################