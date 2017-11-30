TEAM_NAME = "Improved Random"

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

import random
import numpy

visitedCells = []

def randomMove () :
    moves = [MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP]
    return moves[random.randint(0, 3)]

def moveFromLocations (sourceLocation, targetLocation) : 
    difference = tuple(numpy.subtract(targetLocation, sourceLocation))
    if difference == (0, -1) :
        return MOVE_DOWN
    elif difference == (0, 1) :
        return MOVE_UP
    elif difference == (1, 0) :
        return MOVE_RIGHT
    elif difference == (-1, 0) :
        return MOVE_LEFT
    else :
        raise Exception("Impossible move")

def listDiscoveryMoves (playerLocation, mazeMap) :
    moves = []
    for neighbor in mazeMap[playerLocation] :
        if neighbor not in visitedCells :
            move = moveFromLocations(playerLocation, neighbor)
            moves.append(move)
    return moves

def preprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed) :
    pass

def turn (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed) :

    # v-- new code (the current cell is now visited)
    global visitedCells
    if playerLocation not in visitedCells :
        visitedCells.append(playerLocation)
    
    # v-- new code (we get the moves leading to a new cell and apply one of them at random if possible)
    moves = listDiscoveryMoves(playerLocation, mazeMap)
    if moves :
        return moves[random.randint(0, len(moves) - 1)]
    else :
        return randomMove()
