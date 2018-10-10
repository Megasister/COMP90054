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
from heapq import heappop, heappush
from operator import itemgetter
from time import time

from baselineTeam import DefensiveReflexAgent
from captureAgents import CaptureAgent
from game import Directions
from mixins import (
    EvalBaseMixin,
    InferenceMixin,
    OffensiveMixin,
    GreedyDelegate
)


#################
# Team creation #
#################
agent1 = agent2 = None
def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first='DijkstraMonteCarloAgent',
    second='DefensiveReflexAgent'
):
    global agent1, agent2
    agent1 = eval(first)(firstIndex, isRed)
    agent2 = eval(second)(secondIndex, isRed)
    return [agent1, agent2]


################################################################################
# Concrete Agents
################################################################################
class OffensiveGreedyAgent(
    CaptureAgent,
    EvalBaseMixin,
    GreedyDelegate,
    OffensiveMixin,
    InferenceMixin
):
    """
    An agent greedily choose the maximum score of next step only with
    offensive strategy
    """

    def __init__(self, index, red, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        object.__init__(self)

        self._half = self._height = self._maxDist = self._width = None

        self.red = red

        OffensiveMixin.__init__(self)
        InferenceMixin.__init__(self)

        self._teamTotalFood = 0

    def _countFood(self, foods):
        half = self._half
        width = self._width
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
        CaptureAgent.registerInitialState(self, gameState)

        data = gameState.data
        layout = data.layout
        self._height = layout.height
        width = self._width = layout.width
        self._half = width // 2
        self._maxDist = self._height * width

        self._initialiseWeights()
        self._initialise(gameState)

        self._countFood(gameState.data.food.data)

    chooseAction = GreedyDelegate.chooseAction


class DefensiveGreedyAgent(EvalBaseMixin, InferenceMixin):
    """
    An agent greedily choose the maximum score of next step only with
    defensive strategy
    """

    def __init__(self, index, red, timeForComputing=.1):
        EvalBaseMixin.__init__(self, index, red, timeForComputing)
        InferenceMixin.__init__(self)

        self._weights = None

    def registerInitialState(self, gameState):
        """
        Initialise the agent with the given state and initialise a list of
        weights
        """
        EvalBaseMixin.registerInitialState(self, gameState)
        self._initialise(gameState)

        self._weights = [
        ]

    def _eval(self, gameState, action):
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
