####################################################################################################################################################################################################################################
############################################################################################################## LICENSE #############################################################################################################
####################################################################################################################################################################################################################################
#
#    Copyright © 2016 Bastien Pasdeloup (name.surname@gmail.com) and Télécom Bretagne
#
#    This file is part of PyRat.
#
#    PyRat is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyRat is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyRat.  If not, see <http://www.gnu.org/licenses/>.
#
####################################################################################################################################################################################################################################
############################################# PRE-DEFINED CONSTANTS ########################################################################################### CONSTANTES PRÉ-DÉFINIES ############################################
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    In this section, you will find some pre-defined constants that are needed for the game                       #    Dans cette section, vous trouvez des constantes pré-définies nécessaires pour la partie                     #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    TEAM_NAME : string                                                                                           #    TEAM_NAME : string                                                                                          #
#    ------------------                                                                                           #    ------------------                                                                                          #
#                                                                                                                 #                                                                                                                #
#        This constant represents your name as a team                                                             #        Cette constante représente le nom de votre équipe                                                       #
#        Please change the default value to a string of your choice                                               #        Veuillez en changer la valeur par une chaîne de caractères de votre choix                               #
#                                                                                                                 #                                                                                                                #
#    MOVE_XXX : char                                                                                              #    MOVE_XXX : char                                                                                             #
#    ---------------                                                                                              #    ---------------                                                                                             #
#                                                                                                                 #                                                                                                                #
#        The four MOVE_XXX constants represent the possible directions where to move                              #        Les quatre constantes MOVE_XXX représentent les directions possibles où se déplacer                     #
#        The "turn" function should always return one of these constants                                          #        La fonction "turn" doit toujours renvoyer l'une d'entre elles                                           #
#        Please do not edit them (any other value will be considered incorrect)                                   #        Merci de ne pas les éditer (toute autre valeur sera considérée comme incorrecte)                        #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################

TEAM_NAME = "Backtracking"

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

####################################################################################################################################################################################################################################
########################################### SPACE FOR FREE EXPRESSION ######################################################################################### ZONE D'EXPRESSION LIBRE ############################################
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    In this file, you will two functions: "preprocessing" and "turn"                                             #    Dans ce fichier, vous trouverez deux fonctions : "preprocessing" et "turn"                                  #
#    You need to edit these functions to create your PyRat program                                                #    Vous devez éditer ces fonctions pour réaliser votre programme PyRat                                         #
#    However, you are not limited to them, and you can write any Python code in this file                         #    Toutefois, vous n'êtes pas limité(e), et vous pouvez écrire n'importe quel code Python dans ce fichier      #
#    Please use the following space to write your additional constants, variables, functions...                   #    Merci d'utiliser l'espace ci-dessous pour écrire vos constantes, variables, fonctions...                    #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################
# Useful imports #
##################

import heapq
import operator

####################################################################################################################################################################################################################################
# Transforms a pair of locations into a move going from the first to the second #
#################################################################################

def locationsToMove (location1, location2) :
    
    # Depends on the difference
    difference = (location2[0] - location1[0], location2[1] - location1[1])
    if difference == (-1, 0) :
        return MOVE_LEFT
    elif difference == (1, 0) :
        return MOVE_RIGHT
    elif difference == (0, 1) :
        return MOVE_UP
    elif difference == (0, -1) :
        return MOVE_DOWN
    else :
        raise Exception("Invalid location provided")

####################################################################################################################################################################################################################################
# Dijkstra's algorithm to compute the shortest paths from the initial locations to all the nodes #
# Returns for each location the previous one that leads to the shortest path                     #
##################################################################################################

def dijkstra (mazeMap, initialLocation) :
    
    # We initialize the min-heap with the source node
    # Distances and routes are updated once the nodes are visited for the first time
    # The temporary values are stored in the min-heap
    minHeap = [(0, initialLocation, None)]
    distances = {}
    routes = {}
    
    # Main loop
    while len(minHeap) != 0 :
        (distance, location, predecessor) = heapq.heappop(minHeap)
        if location not in distances :
            distances[location] = distance
            routes[location] = predecessor
            for neighbor in mazeMap[location] :
                newDistanceToNeighbor = distance + mazeMap[location][neighbor]
                heapq.heappush(minHeap, (newDistanceToNeighbor, neighbor, location))
    
    # Result
    return (routes, distances)
    
####################################################################################################################################################################################################################################
# Takes as an input the result of Dijkstra's algorithm        #
# Returns the sequence of nodes from sourceNode to targetNode #
###############################################################

def routesToPath (routes, sourceNode, targetNode) :
    
    # Recursive reconstruction
    if sourceNode == targetNode :
        return [sourceNode]
    else :
        return routesToPath(routes, sourceNode, routes[targetNode]) + [targetNode]

####################################################################################################################################################################################################################################
# Returns the sequence of moves in the maze associated to a path in the graph #
###############################################################################

def pathToMoves (path) :
    
    # Recursive reconstruction
    if len(path) <= 1 :
        return []
    else :
        return [locationsToMove(path[0], path[1])] + pathToMoves(path[1:])

####################################################################################################################################################################################################################################
# Returns the sequence of moves in the maze associated to a path in the meta-graph #
####################################################################################

def metaPathToMoves (metaPath, movesOnGraph) :
    
    # Recursive reconstruction
    if len(metaPath) <= 1 :
        return []
    else :
        return movesOnGraph[metaPath[0]][metaPath[1]] + metaPathToMoves(metaPath[1:], movesOnGraph)

####################################################################################################################################################################################################################################
# Returns the meta-graph using the given subset of locations as nodes #
# The edges weights are the shortest path lengths linking the nodes   #
# Returns also the correspondance in terms of moves                   #
#######################################################################
    
def mazeMapToMetaGraph (mazeMap, locationsOfInterest) :
    
    # We perform Dijkstra's algorithm from every node to find the shortest path to others
    metaGraph = {}
    movesOnGraph = {}
    for node1 in locationsOfInterest :
        
        # Dijkstra from this node
        metaGraph[node1] = {}
        movesOnGraph[node1] = {}
        (routes, distances) = dijkstra(mazeMap, node1)
        
        # We store the path and distance to other nodes
        for node2 in locationsOfInterest :
            if node1 != node2 :
                metaGraph[node1][node2] = distances[node2]
                movesOnGraph[node1][node2] = pathToMoves(routesToPath(routes, node1, node2))
    
    # Done
    return (metaGraph, movesOnGraph)

####################################################################################################################################################################################################################################
# This function returns the shortest path in the graph going through all nodes #
################################################################################

def travelingSalesmanWithBacktracking (graph, sourceNode) :

    # We store the best path here
    bestLength = float("inf")
    bestPath = None

    # Thus function takes as an argument the nodes that are not visited yet, the graph and a current location
    # In addition, we remember the currently crossed path and the associated weight
    # Basically, we perform a depth-first search and study the path length if it contains all nodes
    def exhaustive (remainingNodes, currentNode, currentPath, currentLength, graph) :
        
        # If no nodes are remaining, we have a path comprising all nodes
        # We save it as the best if it is better than the current best
        if not remainingNodes :
            nonlocal bestLength, bestPath
            if currentLength < bestLength :
                bestLength = currentLength
                bestPath = currentPath
        
        # If some nodes are remaining, we perform a depth-first search
        # We increase the path and its length in the recursive call
        # Obviously, we only consider nodes that are reachable
        # We ignore nodes that extend the path to a length higher than the current best
        # Also, neighbors are visited in increasing distance order to favor the backtracking
        else :
            sortedNeighbors = sorted(graph[currentNode].keys(), key=operator.itemgetter(1))
            for neighbor in sortedNeighbors :
                if neighbor in remainingNodes :
                    lengthWithNeighbor = currentLength + graph[currentNode][neighbor]
                    if lengthWithNeighbor >= bestLength :
                        continue
                    otherNodes = [x for x in remainingNodes if x != neighbor]
                    exhaustive(otherNodes, neighbor, currentPath + [neighbor], lengthWithNeighbor, graph)
    
    # We initiate the search from the source node
    otherNodes = [node for node in graph if node != sourceNode]
    exhaustive(otherNodes, sourceNode, [sourceNode], 0, graph)
    
    # We return the result
    return (bestPath, bestLength)

####################################################################################################################################################################################################################################
# Global variable containing the result moves to apply #
########################################################

resultMoves = []

####################################################################################################################################################################################################################################
############################################# PREPROCESSING FUNCTION ######################################################################################### FONCTION DE PRÉ-TRAITEMENT ##########################################
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    This function is executed once at the very beginning of the game                                             #    Cette fonction est exécutée une unique fois au tout début de la partie                                      #
#    It allows you to make some computations before the players are allowed to move                               #    Vous pouvez y effectuer des calculs avant que les joueurs ne puissent commencer à bouger                    #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    mazeMap : dict(pair(int, int), dict(pair(int, int), int))                                                    #    mazeMap : dict(pair(int, int), dict(pair(int, int), int))                                                   #
#    ---------------------------------------------------------                                                    #    ---------------------------------------------------------                                                   #
#                                                                                                                 #                                                                                                                #
#        Map of the maze as a data structure                                                                      #        Structure de données représentant la carte du labyrinthe                                                #
#        mazeMap[x] gives you the neighbors of cell x                                                             #        mazeMap[x] renvoie les voisins de la case x                                                             #
#        mazeMap[x][y] gives you the weight of the edge linking cells x and y                                     #        mazeMap[x][y] renvoie le poids de l'arête reliant les cases x et y                                      #
#        if mazeMap[x][y] is undefined, there is no edge between cells x and y                                    #        Si mazeMap[x][y] n'est pas défini, les cases x et y ne sont pas reliées par une arête                   #
#                                                                                                                 #                                                                                                                #
#    mazeWidth : int                                                                                              #    mazeWidth : int                                                                                             #
#    ---------------                                                                                              #    ---------------                                                                                             #
#                                                                                                                 #                                                                                                                #
#        Width of the maze, in number of cells                                                                    #        Largeur du labyrinthe, en nombre de cases                                                               #
#                                                                                                                 #                                                                                                                #
#    mazeHeight : int                                                                                             #    mazeHeight : int                                                                                            #
#    ----------------                                                                                             #    ----------------                                                                                            #
#                                                                                                                 #                                                                                                                #
#        Height of the maze, in number of cells                                                                   #        Hauteur du labyrinthe, en nombre de cases                                                               #
#                                                                                                                 #                                                                                                                #
#    playerLocation : pair(int, int)                                                                              #    playerLocation : pair(int, int)                                                                             #
#    -------------------------------                                                                              #    -------------------------------                                                                             #
#                                                                                                                 #                                                                                                                #
#        Initial location of your character in the maze                                                           #        Emplacement initial de votre personnage dans le labyrinthe                                              #
#        It is a pair (line, column), with (0, 0) being the top-left cell in the maze                             #        C'est une paire (ligne, colonne), (0, 0) étant la case en haut à gauche du labyrinthe                   #
#        playerLocation[0] gives you your current line                                                            #        playerLocation[0] renvoie votre ligne actuelle                                                          #
#        playerLocation[1] gives you your current column                                                          #        playerLocation[1] renvoie votre colonne actuelle                                                        #
#        mazeMap[playerLocation] gives you the cells you can access directly                                      #        mazeMap[playerLocation] renvoie les cases auxquelles vous pouvez accéder directement                    #
#                                                                                                                 #                                                                                                                #
#    opponentLocation : pair(int, int)                                                                            #    opponentLocation : pair(int, int)                                                                           #
#    ---------------------------------                                                                            #    ---------------------------------                                                                           #
#                                                                                                                 #                                                                                                                #
#        Initial location of your opponent's character in the maze                                                #        Emplacement initial du personnage de votre adversaire dans le labyrinthe                                #
#        The opponent's location can be used just as playerLocation                                               #        La position de l'adversaire peut être utilisée comme pour playerLocation                                #
#        If you are playing in single-player mode, this variable is undefined                                     #        Si vous jouez en mode un joueur, cette variable n'est pas définie                                       #
#                                                                                                                 #                                                                                                                #
#    piecesOfCheese : list(pair(int, int))                                                                        #    piecesOfCheese : list(pair(int, int))                                                                       #
#    -------------------------------------                                                                        #    -------------------------------------                                                                       #
#                                                                                                                 #                                                                                                                #
#        Locations of all pieces of cheese in the maze                                                            #        Emplacements des morceaux de fromage dans le labyrinthe                                                 #
#        The locations are given in no particular order                                                           #        Les emplacements sont données dans un ordre quelconque                                                  #
#        As for the players locations, these locations are pairs (line, column)                                   #        Comme pour les positions des joueurs, ces emplacements sont des paires (ligne, colonne)                 #
#                                                                                                                 #                                                                                                                #
#    timeAllowed : float                                                                                          #    timeAllowed : float                                                                                         #
#    -------------------                                                                                          #    -------------------                                                                                         #
#                                                                                                                 #                                                                                                                #
#        Time that is allowed for preprocessing, in seconds                                                       #        Temps alloué pour le pré-traitement, en secondes                                                        #
#        After this time, players will have the right to move                                                     #        Après ce temps, les joueurs pourront commencer à bouger                                                 #
#        If your preprocessing is too long, you will still finish it                                              #        Si votre pré-traitement est trop long, vous terminerez quand même son exécution                         #
#        However, it will not prevent your opponent from moving                                                   #        Toutefois, cela n'empêchera pas votre adversaire de bouger                                              #
#        Make sure to finish your preprocessing within the allowed time                                           #        Assurez vous de terminer votre pré-traitement dans le temps imparti                                     #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    This function does not return anything                                                                       #    Cette fonction ne renvoie rien                                                                              #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################

def preprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed) :
    
    # We create a meta-graph using the pieces of cheese and the initial location
    locationsOfInterest = piecesOfCheese + [playerLocation]
    (metaGraph, movesOnGraph) = mazeMapToMetaGraph(mazeMap, locationsOfInterest)
    
    # We solve the traveling salesman problem on the meta-graph
    (bestPath, bestLength) = travelingSalesmanWithBacktracking(metaGraph, playerLocation)

    # Using the associated moves on the graph, we prepare the moves to perform
    global resultMoves
    resultMoves = metaPathToMoves(bestPath, movesOnGraph)

####################################################################################################################################################################################################################################
################################################# TURN FUNCTION ############################################################################################### FONCTION DE TOUR DE JEU ############################################
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    Once the preprocessing is over, the game starts and players can start moving                                 #    Une fois le pré-traitement terminé, la partie démarre et les joueurs peuvent commencer à bouger             #
#    The "turn" function is called at regular times                                                               #    La fonction "turn" est appelée à intervalles réguliers                                                      #
#    You should determine here your next move, given a game configuration                                         #    Vous devez déterminer ici votre prochain mouvement, étant donnée une configuration du jeu                   #
#    This decision will then be applied, and your character will move in the maze                                 #    Cette décision sera ensuite appliquée, et votre personnage se déplacera dans le labyrinthe                  #
#    Then, the "turn" function will be called again with the new game configuration                               #    Ensuite, la fonction "turn" sera appelée à nouveau, avec la nouvelle configuration du jeu                   #
#    This process is repeated until the game is over                                                              #    Ce processus est répété jusqu'à la fin de la partie                                                         #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    mazeMap : dict(pair(int, int), dict(pair(int, int), int))                                                    #    mazeMap : dict(pair(int, int), dict(pair(int, int), int))                                                   #
#    ---------------------------------------------------------                                                    #    ---------------------------------------------------------                                                   #
#                                                                                                                 #                                                                                                                #
#        Same argument as for the "preprocessing" function                                                        #        Même paramètre que pour la fonction "preprocessing"                                                     #
#        The value of mazeMap does not change between two calls of function "turn"                                #        La valeur de mazeMap ne change pas d'un appel à l'autre de la fonction "turn"                           #
#                                                                                                                 #                                                                                                                #
#    mazeWidth : int                                                                                              #    mazeWidth : int                                                                                             #
#    ---------------                                                                                              #    ---------------                                                                                             #
#                                                                                                                 #                                                                                                                #
#        Same argument as for the "preprocessing" function                                                        #        Même paramètre que pour la fonction "preprocessing"                                                     #
#        The value of mazeWidth does not change between two calls of function "turn"                              #        La valeur de mazeWidth ne change pas d'un appel à l'autre de la fonction "turn"                         #
#                                                                                                                 #                                                                                                                #
#    mazeHeight : int                                                                                             #    mazeHeight : int                                                                                            #
#    ----------------                                                                                             #    ----------------                                                                                            #
#                                                                                                                 #                                                                                                                #
#        Same argument as for the "preprocessing" function                                                        #        Même paramètre que pour la fonction "preprocessing"                                                     #
#        The value of mazeHeight does not change between two calls of function "turn"                             #        La valeur de mazeHeight ne change pas d'un appel à l'autre de la fonction "turn"                        #
#                                                                                                                 #                                                                                                                #
#    playerLocation : pair(int, int)                                                                              #    playerLocation : pair(int, int)                                                                             #
#    -------------------------------                                                                              #    -------------------------------                                                                             #
#                                                                                                                 #                                                                                                                #
#        Current location of your character in the maze                                                           #        Emplacement actuel de votre personnage dans le labyrinthe                                               #
#        At the first call of function "turn", it will be your initial location                                   #        Au premier appel de la fonction "turn", ce sera votre emplacement initial                               #
#                                                                                                                 #                                                                                                                #
#    opponentLocation : pair(int, int)                                                                            #    opponentLocation : pair(int, int)                                                                           #
#    ---------------------------------                                                                            #    ---------------------------------                                                                           #
#                                                                                                                 #                                                                                                                #
#        Current location of your opponent's character in the maze                                                #        Emplacement actuel de votre personnage dans le labyrinthe                                               #
#        At the first call of function "turn", it will be your opponent's initial location                        #        Au premier appel de la fonction "turn", ce sera votre emplacement initial                               #
#        If you are playing in single-player mode, this variable is undefined                                     #        Si vous jouez en mode un joueur, cette variable n'est pas définie                                       #
#                                                                                                                 #                                                                                                                #
#    playerScore : float                                                                                          #    playerScore: float                                                                                          #
#    -------------------                                                                                          #    ------------------                                                                                          #
#                                                                                                                 #                                                                                                                #
#        Your current score when the turn begins                                                                  #        Votre score actuel au début du tour                                                                     #
#        It is initialized at 0, and increases by 1 when you eat a piece of cheese                                #        Il est initialisé à 0, et augmente de 1 pour chaque morceau de fromage mangé                            #
#        If you reach the same piece of cheese as your opponent at the same moment, it is worth 0.5 points        #        Si vous arrivez sur le même morceau de fromage que votre adversaire au même moment, il vaut 0.5 points  #
#        If you are playing in single-player mode, this variable is undefined                                     #        Si vous jouez en mode un joueur, cette variable n'est pas définie                                       #
#                                                                                                                 #                                                                                                                #
#    opponentScore : float                                                                                        #    opponentScore: float                                                                                        #
#    ---------------------                                                                                        #    --------------------                                                                                        #
#                                                                                                                 #                                                                                                                #
#        The score of your opponent when the turn begins                                                          #        Le score de votre adversaire au début du tour                                                           #
#                                                                                                                 #                                                                                                                #
#    piecesOfCheese : list(pair(int, int))                                                                        #    piecesOfCheese : list(pair(int, int))                                                                       #
#    -------------------------------------                                                                        #    -------------------------------------                                                                       #
#                                                                                                                 #                                                                                                                #
#        Locations of all remaining pieces of cheese in the maze                                                  #        Emplacements des morceaux de fromage restants dans le labyrinthe                                        #
#        The list is updated at every call of function "turn"                                                     #        La liste est mise à jour à chaque appel de la fonction "turn"                                           #
#                                                                                                                 #                                                                                                                #
#    timeAllowed : float                                                                                          #    timeAllowed : float                                                                                         #
#    -------------------                                                                                          #    -------------------                                                                                         #
#                                                                                                                 #                                                                                                                #
#        Time that is allowed for determining the next move to perform, in seconds                                #        Temps alloué pour le calcul du prochain mouvement, en secondes                                          #
#        After this time, the decided move will be applied                                                        #        Après ce temps, le mouvement choisi sera appliqué                                                       #
#        If you take too much time, you will still finish executing your code, but you will miss the deadline     #        Si vous prenez trop de temps, votre code finira quand-même son excution, mais vous raterez le timing    #
#        Your move will then be considered the next time PyRat checks for players decisions                       #        Votre mouvement sera alors considéré la prochaine fois que PyRat vérifiera les décisions des joueurs    #
#        However, it will not prevent your opponent from moving if he respected the deadline                      #        Toutefois, cela n'empêchera pas votre adversaire de bouger s'il a respecté le timing                    #
#        Make sure to finish your computations within the allowed time                                            #        Assurez vous de terminer vos calculs dans le temps imparti                                              #
#        Also, save some time to ensure that PyRat will receive your decision before the deadline                 #        Aussi, gardez un peu de temps pour garantir que PyRat recevra votre decision avant la fin du temps      #
#        If you are playing in synchronous mode, this variable is undefined                                       #        Si vous jouez en mode synchrone, cette variable n'est pas définie                                       #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################
#                                                                                                                 #                                                                                                                #
#    This function should return one of the following constants: MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP        #    Cette fonction renvoie l'une des constantes suivantes : MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP           #
#    The returned constant represents the move you decide to perform: down, left, right, up                       #    La constante renvoyée représente le mouvement que vous décidez d'effectuer : bas, gauche, droite, haut      #
#    Any other value will be considered incorrect                                                                 #    Toute autre valeur sera considérée comme incorrecte                                                         #
#                                                                                                                 #                                                                                                                #
####################################################################################################################################################################################################################################

def turn (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed) :
    
    # We apply the moves that were stored in the result variable
    move = resultMoves[0]
    del resultMoves[0]
    return move

####################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################
