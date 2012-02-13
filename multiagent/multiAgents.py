# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
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
    return ghostWeight * closestGhostPos + foodCountWeight * foodCount

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def successors(self, gameState, agentInd):
        """
        Returns a list of (gameState, action) one action away for agentInd.
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

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.max_val(gameState, 0)

    def max_val(self, gameState, depth):
        """
        Returns the maximum value attainable from this node, or simply the action if at root
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        if depth == 0:
            stateActions = self.successors(gameState, 0)
            vals = [self.min_val(state, depth, 0) for state, action in stateActions]
            actions = [action for state, action in stateActions]
            maxVal = max(vals)
            return actions[vals.index(maxVal)]
        else:
            v = float("-inf")
            for state, action in self.successors(gameState, 0):
                v = max(v, self.min_val(state, depth, 0))
            return v

    def min_val(self, gameState, depth, ghostInd):
        """
        Returns the minimum value attainable from this node, at depth.
        GHOSTIND keeps track of how many more min nodes we need.
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        if ghostInd > gameState.getNumAgents() - 3: # we are at the last ghost 
            return min([self.max_val(state, depth + 1) for state, action in self.successors(gameState, ghostInd + 1)])
        return min([self.min_val(state, depth, ghostInd + 1) for state, action in self.successors(gameState, ghostInd + 1)])


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.max_val(gameState, 0)

    def max_val(self, gameState, depth):
        """
        Returns the maximum value attainable from this node, or simply the action if at root
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        if depth == 0:
            stateActions = self.successors(gameState, 0)
            vals = [self.exp_val(state, depth, 0) for state, action in stateActions]
            actions = [action for state, action in stateActions]
            maxVal = max(vals)
            return actions[vals.index(maxVal)]
        else:
            v = float("-inf")
            for state, action in self.successors(gameState, 0):
                v = max(v, self.exp_val(state, depth, 0))
            return v

    def exp_val(self, gameState, depth, ghostInd):
        """
        Returns the expected value attainable from this node, at depth.
        GHOSTIND keeps track of how many more exp nodes we need.
        """
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)
        prob = 1/len(gameState.getLegalActions(ghostInd + 1))
        if ghostInd > gameState.getNumAgents() - 3: # we are at the last ghost 
            vals = [self.max_val(state, depth + 1) for state, action in self.successors(gameState, ghostInd + 1)]
            return sum([val * prob for val in vals])
        vals = [self.exp_val(state, depth, ghostInd + 1) for state, action in self.successors(gameState, ghostInd + 1)]
        return sum([val * prob for val in vals])

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    from util import manhattanDistance
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    if 0 not in scaredTimes:
        ghostWeight = 0
        foodCountWeight = 1
    else:
        ghostWeight = 0.7
        foodCountWeight = .3
    closestGhostPos = min([manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates])
    foodList = food.asList()
    foodCount = food.height*food.width - len(foodList)
    return ghostWeight * closestGhostPos + foodCountWeight * foodCount

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

