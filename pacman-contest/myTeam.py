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


from abc import ABCMeta, abstractmethod
from captureAgents import CaptureAgent
from game import Directions
from operator import itemgetter
from util import nearestPoint
import random
import time
import util


#################
# Team creation #
#################
def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first='offensiveAgent',
    second='atLeastDefensiveAgent'
):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
# force new style Python 2 class by multiple inheritance
class FeatureWeightAgent(object, CaptureAgent):
    """
    This is an abstract class generalising all agents make decision based on
    evaluation on handcrafted features and weights
    """
    __metaclass__ = ABCMeta
    # prohibit creation of __dict__ for fast access
    __slots__ = ()

    def evaluate(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        # return dot product of feature vector and weight vector
        return sum(f * w for f, w in zip(
            self.getFeatures(successor), self.getWeights(successor)
        ))

    def chooseAction(self, gameState):
        # greedy strategy
        return max((
            (a, self.evaluate(gameState, a))
            for a in gameState.getLegalActions(self.index)
        ), itemgetter(1))[0]

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        return successor if pos == nearestPoint(pos) \
            else successor.generateSuccessor(self.index, action)

    @abstractmethod
    def getFeatures(self, gameState):
        """
        an iterable (preferably a list) of features to be used in evaluation
        """
        pass

    @abstractmethod
    def getWeights(self, gameState):
        """
        an iterable (preferably a list) of weights which needs to have same
        number of items as the features return by getFeatures
        """
        pass


class GreedyAgent(FeatureWeightAgent):
    __slots__ = ()

    def getFeatures(self, gameState):
        pass

    def getWeights(self, gameState):
        pass


class AdversialAgent(GreedyAgent):
    __slots__ = ()

    def chooseAction(self, gameState):
        pass


class offensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):

        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)
        self.myCapsules = self.getCapsules(gameState)
        self.foodEaten = 0
        self.final_food_reward = 1
        self.final_food_discount_rate = 0.5
        walls = gameState.getWalls()[:]
        self.width = len(walls)
        self.height = len(walls[0])
        self.initialPosition = gameState.getInitialAgentPosition(self.index)
        self.firstTime = True
        self.ghostHunterStep = 10
        self.toHomeDistance = util.Counter()
        node = util.Queue()
        half_width = self.width / 2
        if not self.red:
            for x in xrange(half_width+1,self.width):
                for y in xrange(1,self.height):
                    if not walls[x][y]:
                        self.toHomeDistance[(x, y)] = -1
            for y in xrange(0,self.height):
                if not walls[half_width][y]:
                    self.toHomeDistance[(half_width,y)] = 1
                    node.push((half_width ,y, 1))
            while not node.isEmpty():
                temp_node = node.pop()
                if not walls[temp_node[0] + 1][temp_node[1]]:
                    if self.toHomeDistance[(temp_node[0] + 1, temp_node[1])] == 0:
                        self.toHomeDistance[(temp_node[0] + 1, temp_node[1])] = temp_node[2] + 1
                        node.push((temp_node[0] + 1, temp_node[1], temp_node[2] + 1))
                if not walls[temp_node[0]][temp_node[1] + 1]:
                    if self.toHomeDistance[(temp_node[0], temp_node[1] + 1)] == 0:
                        self.toHomeDistance[(temp_node[0], temp_node[1] + 1)] = temp_node[2] + 1
                        node.push((temp_node[0], temp_node[1] + 1, temp_node[2] + 1))
                if not walls[temp_node[0] - 1][temp_node[1]]:
                    if self.toHomeDistance[(temp_node[0] - 1, temp_node[1])] == 0:
                        self.toHomeDistance[(temp_node[0] - 1, temp_node[1])] = temp_node[2] + 1
                        node.push((temp_node[0] - 1, temp_node[1], temp_node[2] + 1))
                if not walls[temp_node[0]][temp_node[1] - 1]:
                    if self.toHomeDistance[(temp_node[0], temp_node[1] - 1)] == 0:
                        self.toHomeDistance[(temp_node[0], temp_node[1] - 1)] = temp_node[2] + 1
                        node.push((temp_node[0], temp_node[1] - 1, temp_node[2] + 1))
        else:
            for x in xrange(1, half_width-1):
                for y in xrange(1,self.height):
                    if not walls[x][y]:
                        self.toHomeDistance[(x, y)] = -1
            for y in xrange(0,self.height):
                if not walls[half_width - 1][y]:
                    self.toHomeDistance[(half_width - 1,y)] = 1
                    node.push((half_width - 1,y, 1))
            while not node.isEmpty():
                temp_node = node.pop()
                if not walls[temp_node[0] + 1][temp_node[1]]:
                    if self.toHomeDistance[(temp_node[0] + 1, temp_node[1])] == 0:
                        self.toHomeDistance[(temp_node[0] + 1, temp_node[1])] = temp_node[2] + 1
                        node.push((temp_node[0] + 1, temp_node[1], temp_node[2] + 1))
                if not walls[temp_node[0]][temp_node[1] + 1]:
                    if self.toHomeDistance[(temp_node[0], temp_node[1] + 1)] == 0:
                        self.toHomeDistance[(temp_node[0], temp_node[1] + 1)] = temp_node[2] + 1
                        node.push((temp_node[0], temp_node[1] + 1, temp_node[2] + 1))
                if not walls[temp_node[0] - 1][temp_node[1]]:
                    if self.toHomeDistance[(temp_node[0] - 1, temp_node[1])] == 0:
                        self.toHomeDistance[(temp_node[0] - 1, temp_node[1])] = temp_node[2] + 1
                        node.push((temp_node[0] - 1, temp_node[1], temp_node[2] + 1))
                if not walls[temp_node[0]][temp_node[1] - 1]:
                    if self.toHomeDistance[(temp_node[0], temp_node[1] - 1)] == 0:
                        self.toHomeDistance[(temp_node[0], temp_node[1] - 1)] = temp_node[2] + 1
                        node.push((temp_node[0], temp_node[1] - 1, temp_node[2] + 1))


        self.initiateFoodValue(gameState)



    def initiateFoodValue(self,gameState):
        #this is for initial food value map
        if not self.ghostHunterStep == 10:
            self.ghostHunterStep += 1
        walls = gameState.getWalls()[:]
        self.foodList = []
        foodgrid = self.getFood(gameState)[:]
        for x in xrange(len(foodgrid)):
            for y in xrange(len(foodgrid[0])):
                if(foodgrid[x][y]):
                    self.foodList.append((x,y))
        foodList = self.foodList
        self.foodValue = util.Counter()
        self.eachfoodValue = {}
        for food in foodList:
            tempfoodValue = util.Counter()
            expendable_food = util.Queue()
            tempfoodValue[(food[0],food[1])]=self.final_food_reward
            expendable_food.push((food[0],food[1],self.final_food_reward))
            while not expendable_food.isEmpty():
                temp_food=expendable_food.pop()
                if not walls[temp_food[0] + 1][temp_food[1]]:
                    if tempfoodValue[(temp_food[0] + 1, temp_food[1])] == 0:
                        tempfoodValue[(temp_food[0] + 1, temp_food[1])] = temp_food[2] *self.final_food_discount_rate
                        expendable_food.push((temp_food[0] + 1, temp_food[1], temp_food[2] *self.final_food_discount_rate))
                if not walls[temp_food[0]][temp_food[1] + 1]:
                    if tempfoodValue[(temp_food[0], temp_food[1] + 1)] == 0:
                        tempfoodValue[(temp_food[0], temp_food[1] + 1)] = temp_food[2] *self.final_food_discount_rate
                        expendable_food.push((temp_food[0], temp_food[1] + 1, temp_food[2] *self.final_food_discount_rate))
                if not walls[temp_food[0] - 1][temp_food[1]]:
                    if tempfoodValue[(temp_food[0] - 1, temp_food[1])] == 0:
                        tempfoodValue[(temp_food[0] - 1, temp_food[1])] = temp_food[2] *self.final_food_discount_rate
                        expendable_food.push((temp_food[0] - 1, temp_food[1], temp_food[2] *self.final_food_discount_rate))
                if not walls[temp_food[0]][temp_food[1] - 1]:
                    if tempfoodValue[(temp_food[0], temp_food[1] - 1)] == 0:
                        tempfoodValue[(temp_food[0], temp_food[1] - 1)] = temp_food[2] *self.final_food_discount_rate
                        expendable_food.push((temp_food[0], temp_food[1] - 1, temp_food[2] *self.final_food_discount_rate))
            self.eachfoodValue[food] = tempfoodValue
            for item in tempfoodValue.items():
                self.foodValue[item[0]] += tempfoodValue[item[0]]


    def chooseAction(self, gameState):
        myPqueue = util.PriorityQueue()
        position = gameState.getAgentPosition(self.index)
        if self.amIGhost(position):
            self.foodEaten = 0
        if self.initialPosition == position and not self.firstTime:
            self.initiateFoodValue(gameState)
        opponentsPosition = self.getOpponentsPosition(gameState)
        foodList = list(self.foodList)
        foodValue = self.foodValue.copy()
        foodEaten = self.foodEaten
        myPqueue.push((gameState,1,None,0,foodList,foodValue,foodEaten,10),1)
        max_value = util.Counter()
        current_state = myPqueue.pop()
        visted = []
        numObser = len(opponentsPosition)
        numGhost = 0
        Tvalue = 0
        for opponent in opponentsPosition:
            if(self.isHeGhost(opponent)):
                numGhost += 1
        if numObser == 0 or numGhost == 0 or self.ghostHunterStep < 10:
            Tvalue = 2
        else:
            Tvalue = 15
        while current_state[1]< Tvalue:
            actions = current_state[0].getLegalActions(self.index)
            tempPosition = current_state[0].getAgentPosition(self.index)
            new_opponentsPosition = list(opponentsPosition)
            tempFoodList = list(current_state[4])
            tempFoodValue = current_state[5].copy()
            tempFoodEaten = current_state[6]
            values = util.Counter()
            for action in actions:
                if action == 'North':
                    if not current_state[1] == 1:
                        for opponentPosition in opponentsPosition:
                            new_opponentsPosition.append(self.upGhostLocation((tempPosition[0],tempPosition[1] + 1),opponentPosition,gameState))
                    values[(action,tempPosition[0],tempPosition[1] + 1)] = self.getValue(tempPosition[0],tempPosition[1] + 1,new_opponentsPosition, gameState,tempFoodValue,tempFoodEaten,current_state[7])
                elif action == 'South':
                    if not current_state[1] == 1:
                        for opponentPosition in opponentsPosition:
                            new_opponentsPosition.append(self.upGhostLocation((tempPosition[0],tempPosition[1] - 1),opponentPosition,gameState))
                    values[(action,tempPosition[0],tempPosition[1] - 1)] = self.getValue(tempPosition[0],tempPosition[1] - 1,new_opponentsPosition, gameState,tempFoodValue,tempFoodEaten,current_state[7])
                elif action == 'West':
                    if not current_state[1] == 1:
                        for opponentPosition in opponentsPosition:
                            new_opponentsPosition.append(self.upGhostLocation((tempPosition[0] - 1 ,tempPosition[1]),opponentPosition,gameState))
                    values[(action,tempPosition[0] - 1,tempPosition[1])] = self.getValue(tempPosition[0] - 1,tempPosition[1],new_opponentsPosition,gameState,tempFoodValue,tempFoodEaten,current_state[7])
                elif action == 'East':
                    if not current_state[1] == 1:
                        for opponentPosition in opponentsPosition:
                            new_opponentsPosition.append(self.upGhostLocation((tempPosition[0] + 1,tempPosition[1]),opponentPosition,gameState))
                    values[(action,tempPosition[0] + 1,tempPosition[1])] = self.getValue(tempPosition[0] + 1,tempPosition[1],new_opponentsPosition,gameState,tempFoodValue,tempFoodEaten,current_state[7])
            i=1
            for key in values.sortedKeys():
                #if (key[1],key[2]) in visted:
                if values[key] == -10000:
                    continue
                if (current_state[0].getAgentPosition(self.index),key[0]) in visted:
                    continue
                visted.append((current_state[0].getAgentPosition(self.index),key[0]))
                tempFoodEaten = self.updateFoodInfo(current_state[0].getAgentPosition(self.index), tempFoodList, tempFoodEaten, tempFoodValue)
                if (key[1],key[2]) in self.myCapsules:
                    temp_step=1
                elif current_state[7] != 10:
                    temp_step = current_state[7]+1
                else:
                    temp_step=10
                if current_state[2] == None:
                    myPqueue.push((current_state[0].generateSuccessor(self.index,key[0]),i+current_state[1],key[0],values[key],tempFoodList,tempFoodValue,tempFoodEaten,temp_step),i+current_state[1])
                else:
                    myPqueue.push((current_state[0].generateSuccessor(self.index,key[0]),i+current_state[1],current_state[2],values[key],tempFoodList,tempFoodValue,tempFoodEaten,temp_step),i+current_state[1])
                #i+=1
            if myPqueue.isEmpty():
                return random.choice(gameState.getLegalActions(self.index))
            current_state=myPqueue.pop()# this is for the last element
            visted.append(current_state[0].getAgentPosition(self.index))
            if max_value[current_state[2]] == 0:
                max_value[current_state[2]] = current_state[3]
            elif max_value[current_state[2]]<current_state[3]:
                max_value[current_state[2]]=current_state[3]
        temp_value=-999
        next_action=None
        while not myPqueue.isEmpty():
            current_state=myPqueue.pop()
            if max_value[current_state[2]]<current_state[3]:
                max_value[current_state[2]]=current_state[3]
            # print current_state[3]
            # if current_state[3]>temp_value:
            #     temp_value=current_state[3]
            #     new_postion = gameState.generateSuccessor(self.index,current_state[2]).getAgentPosition(self.index)
            #     next_action= current_state[2]
        next_action = max_value.argMax()
        new_position = gameState.generateSuccessor(self.index, next_action).getAgentPosition(self.index)
        #self.foodEaten = self.updateFoodInfo(new_postion, self.foodList, self.foodEaten, self.foodValue)
        if (new_position[0],new_position[1]) in self.foodList:
            self.foodList.remove((new_position[0],new_position[1]))
            tempFoodValue2 = self.eachfoodValue[(new_position[0],new_position[1])]
            if self.foodEaten == 0:
                self.foodEaten = 2
            else:
                self.foodEaten *= 2
            for item in tempFoodValue2.items():
                self.foodValue[item[0]] -= tempFoodValue2[item[0]]
        if (new_position[0],new_position[1]) in self.myCapsules:
            self.myCapsules.remove((new_position[0],new_position[1]))
            self.ghostHunterStep = 1
        self.firstTime = False
        return next_action

    def upGhostLocation(self,our_new_position,ghost_old_position,gameState):
        minDistance = util.Counter()
        new_ghost_position = []
        new_ghost_position.append((ghost_old_position[0],ghost_old_position[1]))
        new_ghost_position.append((ghost_old_position[0],ghost_old_position[1] + 1))
        new_ghost_position.append((ghost_old_position[0],ghost_old_position[1] - 1))
        new_ghost_position.append((ghost_old_position[0] + 1,ghost_old_position[1]))
        new_ghost_position.append((ghost_old_position[0] - 1,ghost_old_position[1]))
        for element in new_ghost_position:
            if not gameState.hasWall(element[0],element[1]):
                if self.getMazeDistance(our_new_position,element) == 0:
                    return element
                minDistance[element] = -self.getMazeDistance(our_new_position,element)
        return minDistance.argMax()

    def updateFoodInfo(self,position,foodList,foodEaten,foodValue):
        if (position[0],position[1]) in foodList:
            foodList.remove((position[0],position[1]))
            tempFoodValue2 = self.eachfoodValue[(position[0],position[1])]
            if foodEaten == 0:
                foodEaten = 2
            else:
                foodEaten *= 2
            for item in tempFoodValue2.items():
                foodValue[item[0]] -= tempFoodValue2[item[0]]
        return foodEaten

    def getValue(self,x,y,opponentsPosition,gameState,foodValue,foodEaten,step):
        numObser = len(opponentsPosition)
        numGhost = 0
        for opponent in opponentsPosition:
            if(self.isHeGhost(opponent)):
                numGhost += 1
        if numObser == 0 or numGhost == 0:
            if (x,y) in self.myCapsules:
                value = foodValue[(x,y)] + 1 * foodEaten / float(self.toHomeDistance[(x,y)])
            else:
                value = foodValue[(x,y)] + 1 * foodEaten / float(self.toHomeDistance[(x,y)])
            return value
        if numGhost == 1:
            if self.getMazeDistance(opponentsPosition[0],(x,y)) == 0:
                return -10000 * (self.ghostHunterStep  - 10)
            elif self.doIScareGhost((x,y)) or self.ghostHunterStep < 10:
                ghostAffect = 0
            else:
                ghostAffect = 5.0 * (foodEaten + foodValue[(x, y)]) / (float(self.getMazeDistance(opponentsPosition[0],(x,y))) * 0.1)
            value = foodValue[(x, y)] + 1 * foodEaten / (float(self.toHomeDistance[(x,y)]))  - ghostAffect
            if (x,y) in self.myCapsules:
                value += 2
            return value
        else:
            ghostAffect = 0
            for opponentPosition in opponentsPosition:
                if self.isHeGhost(opponentPosition):
                    if self.getMazeDistance(opponentPosition,(x,y)) == 0:
                        return -10000 * (step  - 10)
                    elif self.doIScareGhost((x,y)) or self.ghostHunterStep < 10:
                        ghostAffect = 0
                        break
                    else:
                        ghostAffect += 5.0 * (foodEaten + foodValue[(x, y)]) / (float(self.getMazeDistance(opponentPosition,(x,y))) * 0.1)
            value = foodValue[(x, y)] + 1 * foodEaten / (float(self.toHomeDistance[(x,y)])) - ghostAffect
            if (x,y) in self.myCapsules:
                value += 2
            return value

    def getOpponentsPosition(self,gameState):
        opponents = self.getOpponents(gameState)
        opponentsPosition = []
        for opponent in opponents:
            position = gameState.getAgentPosition(opponent)
            if position:
                opponentsPosition.append(position)
        return opponentsPosition


    def amIGhost(self,position):
        if self.red:
            if position[0] < (self.width / 2):
                return True
            else:
                return False
        else:
            if position[0] < (self.width / 2):
                return False
            else:
                return True

    def doIScareGhost(self,position):
        if self.red:
            if position[0] < ((self.width / 2) - 1):
                return True
            else:
                return False
        else:
            if position[0] < ((self.width / 2) + 1 ):
                return False
            else:
                return True


    def isHeGhost(self,position):
        if self.red:
            if position[0] < (self.width / 2):
                return False
            else:
                return True
        else:
            if position[0] < (self.width / 2):
                return True
            else:
                return False


    def gotoCloestFood(self,gameState):
        foodList = self.foodList
        position = gameState.getAgentPosition(self.index)
        fooddistance = util.Counter
        for food in foodList:
            fooddistance[food] = -self.getMazeDistance(food,position)
        closetfood = fooddistance.argMax()
        bestaction = None
        actiondistance = util.Counter
        for action in gameState.getLegalActions(self.index):
            if action == 'North':
                newposition = (position[0],position[1] + 1)
                actiondistance[action] = -self.getMazeDistance(closetfood,newposition)
            elif action == 'South':
                newposition = (position[0],position[1] - 1)
                actiondistance[action] = -self.getMazeDistance(closetfood,newposition)
            elif action == 'West':
                newposition = (position[0] - 1,position[1])
                actiondistance[action] = -self.getMazeDistance(closetfood,newposition)
            elif action == 'East':
                newposition = (position[0] + 1,position[1])
                actiondistance[action] = -self.getMazeDistance(closetfood,newposition)
        return actiondistance.argMax()

class atLeastDefensiveAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """
        CaptureAgent.registerInitialState(self, gameState)
        walls = gameState.getWalls()[:]
        self.actionList = util.Queue()
        self.width = len(walls)
        self.height = len(walls[0])
        self.initiateFooddistance(gameState)
        self.center = self.foodValue.argMax()





    def initiateFooddistance(self,gameState):
        #this is for initial food value map
        walls = gameState.getWalls()[:]
        self.foodList = []
        foodgrid = self.getFoodYouAreDefending(gameState)[:]
        for x in xrange(len(foodgrid)):
            for y in xrange(len(foodgrid[0])):
                if(foodgrid[x][y]):
                    self.foodList.append((x,y))
        foodList = self.foodList
        self.foodValue = util.Counter()
        self.eachfoodValue = {}
        self.final_food_reward = 0
        for food in foodList:
            tempfoodValue = util.Counter()
            expendable_food = util.Queue()
            tempfoodValue[(food[0],food[1])]=self.final_food_reward
            expendable_food.push((food[0],food[1],self.final_food_reward))
            while not expendable_food.isEmpty():
                temp_food=expendable_food.pop()
                if not walls[temp_food[0] + 1][temp_food[1]]:
                    if tempfoodValue[(temp_food[0] + 1, temp_food[1])] == 0:
                        tempfoodValue[(temp_food[0] + 1, temp_food[1])] = temp_food[2] - 1
                        expendable_food.push((temp_food[0] + 1, temp_food[1], temp_food[2] - 1))
                if not walls[temp_food[0]][temp_food[1] + 1]:
                    if tempfoodValue[(temp_food[0], temp_food[1] + 1)] == 0:
                        tempfoodValue[(temp_food[0], temp_food[1] + 1)] = temp_food[2] - 1
                        expendable_food.push((temp_food[0], temp_food[1] + 1, temp_food[2] - 1))
                if not walls[temp_food[0] - 1][temp_food[1]]:
                    if tempfoodValue[(temp_food[0] - 1, temp_food[1])] == 0:
                        tempfoodValue[(temp_food[0] - 1, temp_food[1])] = temp_food[2] - 1
                        expendable_food.push((temp_food[0] - 1, temp_food[1], temp_food[2]- 1))
                if not walls[temp_food[0]][temp_food[1] - 1]:
                    if tempfoodValue[(temp_food[0], temp_food[1] - 1)] == 0:
                        tempfoodValue[(temp_food[0], temp_food[1] - 1)] = temp_food[2] - 1
                        expendable_food.push((temp_food[0], temp_food[1] - 1, temp_food[2] - 1))
            self.eachfoodValue[food] = tempfoodValue
            for item in tempfoodValue.items():
                self.foodValue[item[0]] += tempfoodValue[item[0]]

    def updateFoodList(self,gameState):
        self.foodList = []
        foodgrid = self.getFoodYouAreDefending(gameState)[:]
        for x in xrange(len(foodgrid)):
            for y in xrange(len(foodgrid[0])):
                if(foodgrid[x][y]):
                    self.foodList.append((x,y))


    def chooseAction(self, gameState):
      if len(self.generateFoodGridToList(gameState)) > self.foodList:
          self.updateFoodList(gameState)
      opponentsPosition = self.getOpponentsPosition(gameState)
      if len(opponentsPosition) == 0:
          eatenFood = self.checkWhichFoodEated(gameState)
          if eatenFood is None:
            if self.actionList.isEmpty():
                self.generateActionList(gameState,self.center)
            return self.actionList.pop()
          else:
              self.updateFoodList(gameState)
              if self.actionList.isEmpty():
                self.generateActionList(gameState,eatenFood)
              return self.actionList.pop()
      elif len(opponentsPosition) == 1:
          if self.getMazeDistance(opponentsPosition[0],gameState.getAgentPosition(self.index)) < 5:
            self.actionList = util.Queue()
            self.updateFoodList(gameState)
            return self.steptoLocation(gameState,opponentsPosition[0])[0]
          else:
              if self.actionList.isEmpty():
                self.generateActionList(gameState,self.center)
              return self.actionList.pop()
      elif len(opponentsPosition) == 2:
        self.actionList = util.Queue()
        position = gameState.getAgentPosition(self.index)
        if self.getMazeDistance(position,opponentsPosition[0]) <  self.getMazeDistance(position,opponentsPosition[1]):
            self.updateFoodList(gameState)
            return self.steptoLocation(gameState,opponentsPosition[0])[0]
        else:
            self.updateFoodList(gameState)
            return self.steptoLocation(gameState,opponentsPosition[1])[0]

    def getOpponentsPosition(self,gameState):
        opponents = self.getOpponents(gameState)
        opponentsPosition = []
        for opponent in opponents:
            position = gameState.getAgentPosition(opponent)
            if position:
                opponentsPosition.append(position)
        return opponentsPosition

    def checkWhichFoodEated(self,gameState):
         eatenFood = []
         for food in self.foodList:
             if food not in self.generateFoodGridToList(gameState):
                 self.foodList.remove(food)
                 eatenFood.append(food)
         if len(eatenFood) == 1:
             return eatenFood[0]
         else:
            return None

    def generateFoodGridToList(self,gameState):
        foodList = []
        foodgrid = self.getFoodYouAreDefending(gameState)[:]
        for x in xrange(len(foodgrid)):
            for y in xrange(len(foodgrid[0])):
                if(foodgrid[x][y]):
                    foodList.append((x,y))
        return foodList

    def generateActionList(self,gameState,destination):
        self.actionList = util.Queue()
        action, newposition = self.steptoLocation(gameState,destination)
        self.actionList.push(action)
        Successor = gameState
        while newposition != destination:
            Successor = Successor.generateSuccessor(self.index,action)
            action, newposition = self.steptoLocation(Successor,destination)
            self.actionList.push(action)




    def steptoLocation(self,gameState,destination):
        actiondistance = util.Counter()
        position = gameState.getAgentPosition(self.index)
        for action in gameState.getLegalActions(self.index):
            if action == 'North':
                newposition = (position[0],position[1] + 1)
                actiondistance[action] = -self.getMazeDistance(destination,newposition)
            elif action == 'South':
                newposition = (position[0],position[1] - 1)
                actiondistance[action] = -self.getMazeDistance(destination,newposition)
            elif action == 'West':
                if not self.red:
                    if position[0] == self.width/2:
                        continue
                newposition = (position[0] - 1,position[1])
                actiondistance[action] = -self.getMazeDistance(destination,newposition)
            elif action == 'East':
                if self.red:
                    if position[0] == self.width/2 - 1:
                        continue
                newposition = (position[0] + 1,position[1])
                actiondistance[action] = -self.getMazeDistance(destination,newposition)
        if len(actiondistance) == 0:
            return 'Stop',position
        final_action = actiondistance.argMax()
        if final_action == 'North':
            newposition = (position[0],position[1] + 1)
        elif final_action == 'South':
            newposition = (position[0],position[1] - 1)
        elif final_action == 'West':
            newposition = (position[0] - 1,position[1])
        elif final_action == 'East':
            newposition = (position[0] + 1,position[1])
        return final_action,newposition

    def isHeGhost(self,position):
        if self.red:
            if position[0] < (self.width / 2):
                return False
            else:
                return True
        else:
            if position[0] < (self.width / 2):
                return True
            else:
                return False
