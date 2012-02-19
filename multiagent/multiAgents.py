# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import searchAgents
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        """
        from util import manhattanDistance
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        if 0 not in newScaredTimes:
            ghostWeight = 0
            foodCountWeight = 1
        else:
            ghostWeight = 0.7
            foodCountWeight = .3

        closestGhostPos = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        foodList = newFood.asList()
        foodCount = newFood.height*newFood.width - len(foodList)
        val = ghostWeight * closestGhostPos + foodCountWeight * foodCount
        return val
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        pos = successorGameState.getPacmanPosition()
        food = successorGameState.getFood()
        manDistanceToClosestNonScaredGhost = food.height+food.width
        for ghost in successorGameState.getGhostStates():
            manDistanceToClosestNonScaredGhost = min(manDistanceToClosestNonScaredGhost, manhattanDistance(pos, ghost.getPosition()))
        foodList = food.asList()
        foodDist = 0
        if foodList:
            distances = [manhattanDistance(pos, foodCord) for foodCord in foodList]
            avgDistToFood = sum(distances)/len(distances)
            distToClosestFood = min(distances)
            distToFarthestFood = max(distances)
            foodDist = distToClosestFood
        if manDistanceToClosestNonScaredGhost > 3:
            weights = [0, 1000000, 1000]
        else:
            weights = [100000, 0, 0]
        ghostVal = weights[0] * manDistanceToClosestNonScaredGhost
        foodVal = weights[1] * (food.height*food.width - len(foodList))
        foodDistVal = weights[2] * ((food.height + food.width) - foodDist)
        val = (ghostVal + foodVal + foodDistVal)
        return val

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    def successors(self, gameState, agentInd):
        """
        Returns a list of game states one action away for agentInd.
        """
        actions = gameState.getLegalActions(agentInd)
        return [(gameState.generateSuccessor(agentInd, action), action) for action in actions]
        
        
    def terminalTest(self, state, depth):
        """
        Returns true if state is a terminal state or max depth is reached.
        """
        if state.isWin() or state.isLose() or self.depth == depth:
            return True
        return False
        
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        
class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.max_val(gameState, 0)
        
    def max_val(self, gameState, depth):
        """
        Returns the maximum value attainable from this node.
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        v = float("-inf")
        for state, action in self.successors(gameState, 0):
            minVal = self.min_val(state, depth, gameState.getNumAgents() - 1)
            if minVal > v:
                maxAction = action
                v = minVal
        if depth == 0:
            return maxAction  
        return v
        
    def min_val(self, gameState, depth, numGhosts):
        """
        Returns the minimum value attainable from this node.
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        if numGhosts > 1:
            vals = [self.min_val(state, depth, numGhosts - 1) for state, action in self.successors(gameState, numGhosts)]
        else:
            vals = [self.max_val(state, depth + 1) for state, action in self.successors(gameState, numGhosts)]
        return min(vals)

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.pruned_max_val(gameState, 0, float("-inf"), float("inf"))
        
    def pruned_max_val(self, gameState, depth, alpha, beta):
        """
        Returns the maximum value attainable from this node.
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        v = float("-inf")
        for state, action in self.successors(gameState, 0):
            minVal = self.pruned_min_val(state, depth, gameState.getNumAgents() - 1, alpha, beta)
            if minVal > v:
                maxAction = action
                v = minVal
            if v > beta:
                return v
            alpha = max(alpha, v)
        if depth == 0:
            return maxAction  
        return v
        
    def pruned_min_val(self, gameState, depth, numGhosts, alpha, beta):
        """
        Returns the minimum value attainable from this node.
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        if numGhosts > 1:
            vals = [self.pruned_min_val(state, depth, numGhosts - 1, alpha, beta) for state, action in self.successors(gameState, numGhosts)]
            return min(vals)
        else:
            v = float("inf")
            for state, action in self.successors(gameState, numGhosts):
                v = min(v, self.pruned_max_val(state, depth + 1, alpha, beta))
                if v < alpha:
                    return v
                b = min(beta, v)
            return v
            

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.max_val(gameState, 0)
    
    def nthInd(ls, n, x):
        """
        Returns the index of the nth occurrence of x, or -1 if it doensn't exist.
        """
        ret = -1
        removals = 0
        while n > 1:
            i = ls.index(x)
            ls.remove(x)
            removals += 1
    def max_val(self, gameState, depth):
        """
        Returns the maximum value attainable from this node.
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        if depth == 0:
            stateActions = self.successors(gameState, 0)
            vals = [self.exp_val(state, depth, gameState.getNumAgents() - 1) for state, action in stateActions]
            maxVal = max(vals)
            actions = [action for state, action in stateActions]
            action = actions[vals.index(maxVal)]
            removed = 0
            while vals.count(maxVal) > 1 and action == Directions.STOP:
                vals.remove(maxVal)
                removed += 1
                action = actions[vals.index(maxVal) + removed]
            return action
        else:
            v = float("-inf")
            for state, action in self.successors(gameState, 0):
                v = max(v, self.exp_val(state, depth, gameState.getNumAgents() - 1))
            return v
            
    def exp_val(self, gameState, depth, numGhosts):
        """
        Returns the minimum value attainable from this node.
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        if numGhosts > 1:
            vals = [self.exp_val(state, depth, numGhosts - 1) for state, action in self.successors(gameState, numGhosts)]
        else:
            vals = [self.max_val(state, depth + 1) for state, action in self.successors(gameState, numGhosts)]
        return sum(vals)/len(vals)
    

def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).
        DESCRIPTION: either goes for food or runs away
    """
    pos = currentGameState.getPacmanPosition()  
    food = currentGameState.getFood()
    manDistanceToClosestPower = food.height+food.width
    for capsule in currentGameState.getCapsules():
        manDistanceToClosestPower = min(manDistanceToClosestPower, manhattanDistance(pos, capsule))
    manDistanceToClosestNonScaredGhost = food.height+food.width
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer <= 0:
            manDistanceToClosestNonScaredGhost = min(manDistanceToClosestNonScaredGhost, manhattanDistance(pos, ghost.getPosition()))
    foodList = food.asList()
    foodDist = 0
    if foodList:
        foodDist = searchAgents.closestFoodMazeDist(currentGameState)#(.3*avgDistToFood + .6*distToClosestFood + .1*distToFarthestFood)
    if manDistanceToClosestNonScaredGhost > 3:
        weights = [0, 10000, 1000, 400]
    else:
        weights = [10000, 0, 0, 1]
    ghostVal = weights[0] * manDistanceToClosestNonScaredGhost
    foodVal = weights[1] * (food.height*food.width - len(foodList))
    foodDistVal = weights[2] * ((food.height*food.width) - foodDist)
    powerVal = weights[3] * (food.height*food.width) - manDistanceToClosestPower
    val = (ghostVal + foodVal + foodDistVal)
    return val

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
    """
    def __init__(self, evalFn='contest'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = 3

    def getAction(self, gameState):
        return self.pruned_max_val(gameState, 0, float("-inf"), float("inf"))
        
    def pruned_max_val(self, gameState, depth, alpha, beta):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        v = float("-inf")
        cap = float("inf")
        for state, action in self.successors(gameState, 0):
            minVal = self.pruned_min_val(state, depth, gameState.getNumAgents() - 1, alpha, beta)
            if minVal > v:
                maxAction = action
                v = minVal
            #if depth == 0 and minVal == v and action != Directions.STOP and maxAction == Directions.STOP:
            #    maxAction  
            if v > beta:
                return v
            alpha = max(alpha, v)
        if depth == 0:
            return maxAction  
        return v

    def pruned_min_val(self, gameState, depth, numGhosts, alpha, beta):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        if numGhosts > 1:
            vals = [self.pruned_min_val(state, depth, numGhosts - 1, alpha, beta) for state, action in self.successors(gameState, numGhosts)]
            return min(vals)
        else:
            v = float("inf")
            for state, action in self.successors(gameState, numGhosts):
                v = min(v, self.pruned_max_val(state, depth + 1, alpha, beta))
                if v < alpha:
                    return v
                b = min(beta, v)
            return v
            
def contest(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).
        DESCRIPTION: either goes for food or runs away
    """
    pos = currentGameState.getPacmanPosition()  
    food = currentGameState.getFood()
    manDistanceToClosestPower = food.height+food.width
    for capsule in currentGameState.getCapsules():
        manDistanceToClosestPower = min(manDistanceToClosestPower, manhattanDistance(pos, capsule))
    manDistanceToClosestNonScaredGhost = food.height+food.width
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer <= 0:
            manDistanceToClosestNonScaredGhost = min(manDistanceToClosestNonScaredGhost, manhattanDistance(pos, ghost.getPosition()))
    foodList = food.asList()
    foodDist = 0
    if foodList:
        foodDist = searchAgents.closestFoodMazeDist(currentGameState)#(.3*avgDistToFood + .6*distToClosestFood + .1*distToFarthestFood)
    if manDistanceToClosestNonScaredGhost > 3:
        weights = [0, 10000, 1000, 400]
    else:
        weights = [10000, 0, 0, 1]
    ghostVal = weights[0] * manDistanceToClosestNonScaredGhost
    foodVal = weights[1] * (food.height*food.width - len(foodList))
    foodDistVal = weights[2] * ((food.height*food.width) - foodDist)
    powerVal = weights[3] * (food.height*food.width) - manDistanceToClosestPower
    val = (ghostVal + foodVal + foodDistVal)
    return val
    
def saba(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).
        DESCRIPTION: either goes for food or runs away
    """
    import searchAgents
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    manDistanceToClosestNonScaredGhost = food.height+food.width
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer <= 0:
            manDistanceToClosestNonScaredGhost = min(manDistanceToClosestNonScaredGhost, manhattanDistance(pos, ghost.getPosition()))
    foodList = food.asList()
    foodDist = 0
    if foodList:
        foodDist = searchAgents.closestFoodMazeDist(currentGameState)
    if manDistanceToClosestNonScaredGhost > 3:
        weights = [0, 10000, 1000]
    else:
        weights = [10000, 0, 0]
    ghostVal = weights[0] * manDistanceToClosestNonScaredGhost
    foodVal = weights[1] * (food.height*food.width - len(foodList))
    foodDistVal = weights[2] * ((food.height*food.width) - foodDist)
    val = (ghostVal + foodVal + foodDistVal)
    return val
