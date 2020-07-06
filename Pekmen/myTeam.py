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

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
import game
from util import nearestPoint

trenutnaPartija = 0
brojTreninga = 0
prethodnoStanje = (0, 0)
vremeUplasenogDuha = 0
alfa = 0.8
epsilon = 0.5
gama = 0.8

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveApproximateQAgent', second = 'DefensiveApproximateQAgent', numTraining=0, **args):
  global brojTreninga
  global alfa
  global epsilon
  brojTreninga = numTraining
  if brojTreninga == 0:
    alfa = 0
    epsilon = 0
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.weights = {
      '#-of-food-carrying': -1.0, 
      'successor-score': 35.0, 
      'get-capsule': -10.0, 
      '#-of-ghosts-1-step-away': 3.0, 
      'closest-food': -10.0,
      'back': 20.0, 
      'reverse': -6.0,
      'eats-ghost': -20.0
    }
    self.gamma = 0.8
    self.reward = 0
    
  def getAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successor-score'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successor-score': 1.0}


class OffensiveApproximateQAgent(ReflexCaptureAgent):

  def getQValue(self, gameState, action):
    features = self.getFeatures(gameState, action)
    value = 0.0
    for feature in features:
      value += features[feature] * self.weights[feature]
    return value

  def computeValueFromQValues(self, state):
    possibleStateQValues = util.Counter()
    for action in state.getLegalActions(self.index):
      possibleStateQValues[action] = self.getQValue(state, action)

    if len(possibleStateQValues) > 0:
      return possibleStateQValues[possibleStateQValues.argMax()]
    return 0.0

  def computeActionFromQValues(self, state):
    possibleStateQValues = util.Counter()
    possibleActions = state.getLegalActions(self.index)
    if "Stop" in possibleActions: possibleActions.remove("Stop")
    if len(possibleActions) == 0:
      return None

    for action in possibleActions:
      possibleStateQValues[action] = self.getQValue(state, action)

    best_actions = []
    best_value = possibleStateQValues[possibleStateQValues.argMax()]

    for action, value in possibleStateQValues.items():
      if value == best_value:
        best_actions.append(action)

    return random.choice(best_actions)

  def getPolicy(self, gameState):
    return  self.computeActionFromQValues(gameState)
    
  def getAction(self, gameState):
    legalActions = gameState.getLegalActions(self.index)
    action = None
    
    if len(legalActions) > 0:
      if util.flipCoin(epsilon):
        action = random.choice(legalActions)
      else:
        action = self.getPolicy(gameState)

    nextState = self.getSuccessor(gameState, action)
    reward = self.computeReward(gameState, action)
    self.update(gameState, action, nextState, reward)
    return action

  def computeReward(self, gameState, action):
    nextState = self.getSuccessor(gameState, action)
    myCurrentPos =  gameState.getAgentState(self.index).getPosition()
    enemies = []
    enemyGhost = []
    closestEnemyGhostDist = 9999
    for opponent in self.getOpponents(gameState):
      enemy = gameState.getAgentState(opponent)
      enemies.append(enemy)
    enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    enemyGhostPosition = [ghost.getPosition() for ghost in enemyGhost]
    if len(enemyGhost) > 0:
      closestEnemyGhostDist = (min([self.getMazeDistance(myCurrentPos, ghostPos) for ghostPos in enemyGhostPosition]))
      closestGhost = [gh for gh in enemyGhost if self.getMazeDistance(myCurrentPos, gh.getPosition()) == closestEnemyGhostDist][0]
    global vremeUplasenogDuha
      
    x, y = gameState.getAgentPosition(self.index)
    dx, dy = Actions.directionToVector(action)
    xAfterMove, yAfterMove = int(x + dx), int(y + dy)

    if gameState.hasFood(xAfterMove, yAfterMove):
      wallCount = 0
      if gameState.hasWall(xAfterMove + 1, yAfterMove):
        wallCount += 1
      if gameState.hasWall(xAfterMove - 1, yAfterMove):
        wallCount += 1
      if gameState.hasWall(xAfterMove, yAfterMove + 1):
        wallCount += 1
      if gameState.hasWall(xAfterMove, yAfterMove - 1):
        wallCount += 1
      # u ovom slucaju postoje velike sanse da ga duh pojede ako udje u slepu ulicu da pojede hranu
      if wallCount >= 3 and closestEnemyGhostDist <= 2 and closestGhost.scaredTimer == 0:
        reward = -10
      else:
        # ako je pojeo hranu dobija nagradu od 2 poena
        reward = 2
    else:
      reward = -1

    # ako je pojeo kapsulu dobije nagradu od 15 poena
    if closestGhost.scaredTimer == 39:
      reward += 15

    global prethodnoStanje
    # ako ga je duh pojeo dobija kaznu od -100 poena
    if (abs(prethodnoStanje[0] - xAfterMove) > 1 or abs(prethodnoStanje[1] - yAfterMove) > 1) and prethodnoStanje != (0, 0):
      reward -= 100
    prethodnoStanje = (xAfterMove, yAfterMove)

    # ako je pojeo duha dobija nagradu od 20 poena
    if abs(vremeUplasenogDuha - closestGhost.scaredTimer) > 2:
      reward += 20
    vremeUplasenogDuha = closestGhost.scaredTimer

    # ako je vratio hranu na svoju polovinu bice nagradjen sa 5 poena
    if self.getScore(nextState) - self.getScore(gameState) > 0:
      reward += 5

    return reward

  def update(self, gameState, action, nextState, reward):
    features = self.getFeatures(gameState,action)
    diff = alfa * ((reward + self.gamma * self.computeValueFromQValues(nextState)) - self.getQValue(gameState, action))
    for feature in features:
      self.weights[feature] = self.weights[feature] + diff * features[feature]
      
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    defendingFoodList = self.getFoodYouAreDefending(gameState).asList()
    walls = gameState.getWalls()
    myPosition = successor.getAgentState(self.index).getPosition()
    myCurrentPos =  gameState.getAgentState(self.index).getPosition()
    nextMePosition = successor.getAgentState(self.index).getPosition()
    initPos = gameState.getInitialAgentPosition(self.index)
    currentCarry = gameState.getAgentState(self.index).numCarrying

    mid = gameState.data.layout.width / 2
    if gameState.isOnRedTeam(self.index):
      mid = mid - 1
    else:
      mid = mid + 1

    legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    border = [p for p in legalPositions if p[0] == mid]

    distanceToBorder = min([self.getMazeDistance(myPosition, borderPos) for borderPos in border])

    otherFood = []
    for nFood in foodList:
      foodX, foodY = nFood
      wallCount1 = 0
      if gameState.hasWall(foodX + 1, foodY):
        wallCount1 += 1
      if gameState.hasWall(foodX - 1, foodY):
        wallCount1 += 1
      if gameState.hasWall(foodX, foodY + 1):
        wallCount1 += 1
      if gameState.hasWall(foodX, foodY - 1):
        wallCount1 += 1
      if wallCount1 < 3:
        otherFood.append(nFood)

    capsules = gameState.getCapsules()
    for defCap in self.getCapsulesYouAreDefending(gameState):
      capsules.remove(defCap)

    minDistance = min([self.getMazeDistance(myPosition, food) for food in foodList])
    features['closest-food'] = float(minDistance) / (walls.width * walls.height)
    features['#-of-ghosts-1-step-away'] = 0.0

    blueFood = gameState.getBlueFood().asList()
    redFood = gameState.getRedFood().asList()
    
    if gameState.isOnRedTeam(self.index):      
      if len(blueFood) != 0:
        features['successor-score'] = -float(len(foodList)) / len(blueFood)
    else:
      if len(redFood) != 0:
        features['successor-score'] = -float(len(foodList)) / len(redFood)
   
    enemies = []
    enemyGhost = []
    enemyPacman = []
    for opponent in self.getOpponents(gameState):
      enemy = gameState.getAgentState(opponent)
      enemies.append(enemy)
    enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    enemyPacman = [a for a in enemies if a.isPacman and a.getPosition() != None]
    x, y = gameState.getAgentPosition(self.index)
    dx, dy = Actions.directionToVector(action)
    nextPosition = (int(x + dx), int(y + dy))

    currentX, currentY = myPosition

    ghostPositions = []
    enemiesInvisible = False

    ranges = []
    enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]
    enemyPacmanPosition = [Pacman.getPosition() for Pacman in enemyPacman]

    eatMoreDot = False
    dontEatSecondCap = False
    isScared = False
    escapeDistance = 9999

    if len(enemyGhost) > 0:
      escapeDistance = min([self.getMazeDistance(successor.getAgentPosition(self.index), ghostPosition) for ghostPosition in enemyGhostPosition])
      enemiesInvisible = True
    
    if escapeDistance < 10 or successor.getAgentPosition(self.index) == initPos:
      for ghostNearby in enemyGhost:
        if ghostNearby.scaredTimer > 0:
          dontEatSecondCap = True
        if ghostNearby.scaredTimer > 8:
          eatMoreDot = True
        if ghostNearby.scaredTimer > 30:
          isScared = True
        else:
          isScared = False

      if isScared:
        distanceToGhost = min([self.getMazeDistance(successor.getAgentPosition(self.index), ghostPosition) for ghostPosition in enemyGhostPosition])
        features['eats-ghost'] = float(distanceToGhost * 10) / (walls.width * walls.height)
      else:
        if gameState.getAgentState(self.index).isPacman and not eatMoreDot:
          if len(capsules) > 0 and not dontEatSecondCap:
            distanceToCapsules = min(self.getMazeDistance(successor.getAgentPosition(self.index),capsule) for capsule in capsules)
            features['get-capsule'] = float(distanceToCapsules * 100) / (walls.width * walls.height)
          if currentCarry != 0:
            features['back'] = -float(distanceToBorder) / (walls.width * walls.height)
          
          closestEnemyGhostDist = (min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]))
          if closestEnemyGhostDist == 1:
            features['#-of-ghosts-1-step-away'] = -float(min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition])) / (walls.width * walls.height)
            features['reverse'] = 1
             
          minFoodDistance = min([(self.getMazeDistance(myPosition, food), food) for food in foodList])
              
          if minFoodDistance[0] < 4 and minFoodDistance[1] not in otherFood and closestEnemyGhostDist <= 2 and action != Directions.STOP:
            features['back'] = 0
            wallCount = 0
            newFoodX, newFoodY = minFoodDistance[1]

            if gameState.hasFood(newFoodX + 1, newFoodY):
              dangerFood0 = (newFoodX + 1, newFoodY)
              if dangerFood0 in otherFood:
                otherFood.remove(dangerFood0)
            
            if gameState.hasFood(newFoodX - 1, newFoodY):
              dangerFood1 =(newFoodX - 1, newFoodY)
              if dangerFood1 in otherFood:
                otherFood.remove(dangerFood1)

            if gameState.hasFood(newFoodX,newFoodY + 1):
              dangerFood2 =(newFoodX,newFoodY + 1)
              if dangerFood2 in otherFood:
                otherFood.remove(dangerFood2)
              
            if gameState.hasFood(newFoodX,newFoodY - 1):
              dangerFood3 = (newFoodX,foodY - 1)
              if dangerFood3 in otherFood:
                otherFood.remove(dangerFood3)
            
            if len(otherFood) > 0:
              minOtherFoodDistance = min([(self.getMazeDistance(myPosition, food), food) for food in otherFood])
              features['#-of-food-carrying'] = float(minOtherFoodDistance[0]) / (walls.width * walls.height)

              if closestEnemyGhostDist >= 2:
                features['#-of-ghosts-1-step-away'] = 0
                features['reverse'] = 0
              features['closest-food'] = 0
              if gameState.isOnRedTeam(self.index):
                blueFood.remove(minFoodDistance[1])
                if len(blueFood) != 0:
                  features['successor-score'] = -float(len(otherFood)) / len(blueFood)
              else:
                redFood.remove(minFoodDistance[1])
                if len(redFood) != 0:
                  features['successor-score'] = -float(len(otherFood)) / len(redFood)               
        else:
          if eatMoreDot:
            pass
          else:
            if len(enemyGhostPosition) > 0:          
              if min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]) <= 2:
                features['back'] = -float(distanceToBorder) / (walls.width * walls.height)
                features['#-of-ghosts-1-step-away'] = -float(min(self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition))/(walls.width * walls.height)
                rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
                if action == rev:
                  features['reverse'] = 1
          
    if not eatMoreDot:
      if len(foodList) <= 2 or currentCarry >= 3:
        features['closest-food'] = 0
        features['successor-score'] = 0
        isScared1 = False
        features['back'] = -float(distanceToBorder) / (walls.width * walls.height)

        if len(enemyGhost) != 0 and not dontEatSecondCap:
          if (min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition])) == 1:
            if len(capsules) > 0:
              distanceToCapsules = min(self.getMazeDistance(successor.getAgentPosition(self.index), capsule) for capsule in capsules)
              features['get-capsule'] = float(distanceToCapsules) / (walls.width * walls.height)
            
            features['#-of-ghosts-1-step-away'] = -float(min(self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition))/(walls.width * walls.height)
            rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if action == rev:
              features['reverse'] = 1
      
    features.divideAll(10.0)
    return features

  def final(self, state):
    global brojTreninga
    global alfa
    global epsilon
    global trenutnaPartija
    trenutnaPartija += 1
    print("Zavrsena partija broj: {}; alfa: {}; epsilon: {}\n-------------------------".format(trenutnaPartija, alfa, epsilon))

    if brojTreninga == 0:
      alfa = 0
      epsilon = 0
    else:
      alfa = (1 - ((trenutnaPartija) / brojTreninga)) * alfa
      epsilon = (1 - ((trenutnaPartija) / brojTreninga)) * epsilon

    if brojTreninga == 0 or trenutnaPartija >= brojTreninga: 
      epsilon = 0
      alfa = 0

    f = open("tezine.txt", "w")
    f.write(str(self.weights))
    f.close()
    
  
class DefensiveApproximateQAgent(ReflexCaptureAgent):

  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.defendingFood = []
    self.target = ()
    self.isTargetToFood = False

  def getBorder(self,gameState):
    mid = gameState.data.layout.width/2

    if self.red:
      mid = mid - 1
    else:
      mid = mid + 1

    legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    border = [p for p in legalPositions if p[0] == mid]
    return border
  
  def getWeights(self, gameState, action):
    return { 
      'onDefense': 100, 
      'invaderDist': -10, 
      'nearDist': -10, 
      'reverse': -10,
      'numInvaders': -1000
    }
  
  def getAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    myCurrentPos = gameState.getAgentState(self.index).getPosition()
    initPos = gameState.getInitialAgentPosition(self.index)
    actions.remove(Directions.STOP)

    if gameState.getAgentState(self.index).scaredTimer == 0:
      for a in actions:
        successor = self.getSuccessor(gameState, a)
        nextState = successor.getAgentState(self.index)
        if nextState.isPacman:
          actions.remove(a)

    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction              

    return random.choice(bestActions)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    nextState = successor.getAgentState(self.index)
    nextPosition = nextState.getPosition()
    currentPosition = gameState.getAgentState(self.index).getPosition()
    myFood = self.getFoodYouAreDefending(gameState).asList()
    borderPos = random.choice(self.getBorder(gameState))

    # lociraj pojedenu tacku
    if len(self.defendingFood) > len(myFood):
      self.target = list(set(self.defendingFood) - set(myFood))[0]
      self.defendingFood = myFood
      self.isTargetToFood = True
    
    # kad se ispovraca
    if len(self.defendingFood) < len(myFood):
      self.defendingFood = myFood
    
    features['onDefense'] = 1
    if nextState.isPacman: features['onDefense'] = 0

    opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    nearOpponents = [i for i in opponents if (not i.isPacman) and (i.getPosition() != None)]
    invaders = [i for i in opponents if i.isPacman and i.getPosition() != None]
    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      invaderDist = [self.getMazeDistance(nextPosition, a.getPosition()) for a in invaders]
      features['invaderDist'] = min(invaderDist)

      invaderDistClose = [self.getMazeDistance(currentPosition, a.getPosition()) for a in invaders]

      # ako je uplasen treba da bezi
      if (min(invaderDistClose) == 1 or min(invaderDist) == 1) and gameState.getAgentState(self.index).scaredTimer > 0:
        features['onDefense'] = 0
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == reverse: features['reverse'] = 1
          
      self.target = borderPos
      self.isTargetToFood = False
    if len(invaders) == 0:
      # ako nema pakmana na nasoj polovini i nije pojedena hrana, drzi se blizu protivnickih duhova
      if len(nearOpponents) > 0 and self.isTargetToFood == False:
        nearDist = [self.getMazeDistance(nextPosition, a.getPosition()) for a in nearOpponents]
        features['nearDist'] = min(nearDist)
  
    return features