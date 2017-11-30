# Template file to create an AI for the game PyRat
# http://formations.telecom-bretagne.eu/pyrat

###############################
# Team name to be displayed in the game 
TEAM_NAME = "ratorch"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

###############################
# Please put your imports here
import numpy as np
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from helpers import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
import os
import os.path
import csv


import matplotlib.pyplot as plt
import seaborn
seaborn.set()

###############################
# Please put your global variables here

# Global variables

global log_every, render_every, count_episodes, env, policy, optimizer, reward_avg, value_avg, scores_avg
global action, actions, value, values, rewards, done, state, states, score, scores, firstAction

mse = nn.MSELoss()

DROP_MAX = 0.3
DROP_MIN = 0.05
DROP_OVER = 15000000
IMPORTANCE = 0.8
GOAL_VALUE = 10
VISIBLE_RADIUS = 1


START_HEALTH = 0
STEP_VALUE = -1.0
STEP_NO_REWARD = -0.2

file_name = 'checkpoint.pth.tar'

# Using REINFORCE with a baseline value

gamma = 0.9 # Discounted reward factor


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# Function to create a numpy array representation of the maze
def input_of_parameters(maze, player, opponent, mazeHeight, mazeWidth, piecesOfCheese, verbose=False):
    im_size = (6,2*mazeHeight-1,2*mazeWidth-1)
    canvas = np.zeros(im_size)
    (x,y) = player
    (xx,yy) = opponent
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (i_q, j_q) in piecesOfCheese:
        canvas[0][center_y+j_q-y][center_x+i_q-x] = 1.0
    for i in range(-mazeWidth+1, mazeWidth):
        for j in range(-mazeHeight+1, mazeHeight):
            correct_i = i+x
            correct_j = j+y
            map_i = center_x+i
            map_j = center_y+j
            if (correct_i, correct_j) == (xx,yy):
                canvas[1][map_j][map_i] = -1.0  
            if (correct_i,correct_j) in maze.keys():
                if not((correct_i-1,correct_j) in maze[(correct_i,correct_j)]): #Mur a gauche
                    canvas[2,map_j,map_i] = -1.0
                    pass
                if not((correct_i+1,correct_j) in maze[(correct_i,correct_j)]): #Mur a droite
                    canvas[3,map_j,map_i] = -1.0
                    pass
                if not((correct_i,correct_j+1) in maze[(correct_i,correct_j)]): #Mur en haut
                    canvas[4,map_j,map_i] = -1.0
                    pass
                if not((correct_i,correct_j-1) in maze[(correct_i,correct_j)]): #Mur en bas
                    canvas[5,map_j,map_i] = -1.0 
    canvas[1][center_y][center_x] = 1.0
    canvas = np.expand_dims(canvas, axis=0)
    x_1 = canvas
    x_2 = canvas[:,:,center_y-1:center_y+2,center_x-1:center_x+2]
    return x_1, x_2

# Grid
class Grid():
    def __init__(self, mazeWidth,mazeHeight):
        self.mazeHeight = mazeHeight
        self.mazeWidth = mazeWidth
        
    def reset(self, mazeMap, playerLocation, opponentLocation, piecesOfCheese):
        self.grid = input_of_parameters(mazeMap, playerLocation, opponentLocation, self.mazeHeight, self.mazeWidth, piecesOfCheese)
    
    def visible(self, pos):
        y, x = pos
        #return self.grid[y-VISIBLE_RADIUS:y+VISIBLE_RADIUS+1, x-VISIBLE_RADIUS:x+VISIBLE_RADIUS+1,:]
        return self.grid

# The Agent

class Agent:
    def reset(self,pos):
        self.health = START_HEALTH
        self.score = 0.0
        self.opScore = 0.0
        self.pos = pos
        self.totalMoves = 0.0
        self.lastAction = None
        self.countBlock = 0.0
        self.negativeReward = 0.0

    def evaluateAction(self, agentLocation, agentScore, opponentScore):
        if agentScore > self.score:
            self.health = 1.0
        else:
            self.health = 0.0
        if self.pos[0] == agentLocation[0] and self.pos[1] == agentLocation[1]:
            self.countBlock += 1.0 	    
        self.totalMoves += 1.0
        self.negativeReward = self.countBlock/self.totalMoves
        self.score = agentScore
        self.opScore = opponentScore
        self.pos = agentLocation
        

    def planAction(self, nextAction):
        self.lastAction = nextAction

class Environment:
    def __init__(self, mazeWidth, mazeHeight):
        self.grid = Grid(mazeWidth, mazeHeight)
        self.agent_player = Agent()
        self.agent_opponent = Agent()

    def reset(self, mazeMap, playerLocation, opponentLocation, piecesOfCheese):
        """Start a new episode by resetting grid and agent"""
        self.grid.reset(mazeMap, playerLocation, opponentLocation, piecesOfCheese)
        self.agent_player.reset(playerLocation)
        self.agent_opponent.reset(opponentLocation)    
        self.t = 0
        self.history = []
        self.record_step()
        return self.visible_state
    
    def record_step(self):
        """Add the current state to history for display later"""
        grid = self.grid.grid
        self.history.append((grid, self.agent_player.lastAction, self.agent_player.health, self.agent_player.score, self.agent_opponent.health, self.agent_opponent.score))
    
    @property
    def visible_state(self):
        """Return the visible area surrounding the agent, and current agent health"""
        input1, input2 = self.grid.grid
        y, x = self.agent_player.pos
        extras  = np.array([y, x, self.agent_player.score, self.agent_player.opScore])
        input2 = np.concatenate((input2.flatten(),extras),0)
        return (input1, input2)


    def get_input_size_visible_state(self):
        vis = visible_state()
        return vis[0].shape, vis[1].shape

    def update_action(self, action):
        self.agent_player.planAction(action)

    def step(self, mazeMap, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, done):
        """Update state (grid and agent) based on an action"""
        self.grid.reset(mazeMap, playerLocation, opponentLocation, piecesOfCheese)
        self.agent_player.evaluateAction(playerLocation, playerScore, opponentScore)
        self.agent_opponent.evaluateAction(opponentLocation, opponentScore, playerScore)
        self.record_step()
        if not done:
            reward = self.agent_player.health # Reward will only come at the end
        elif self.agent_player.score > self.agent_opponent.score:
            reward = 2.0
        else:
            reward = IMPORTANCE*(float(playerScore)/opponentScore) - (1-IMPORTANCE)* self.agent_player.negativeReward
        return self.visible_state, reward, self.agent_player.lastAction, self.agent_player.score
	
class ConvGridNet(nn.Module):
    def __init__(self):
        super(ConvGridNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_drop = nn.Dropout2d()
        self.fc_player = nn.Linear(58, 20)
        self.fc1 = nn.Linear(3860, 512)

    def forward(self, input_grid, input_player):
        c1 = F.relu(F.max_pool2d(self.conv1(input_grid),2))
        c2 = F.relu(F.max_pool2d(self.conv2(c1),2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(c2)), 2))
        x = x.view(input_grid.size(0), -1)
        y = self.fc_player(input_player.view(input_grid.size(0),-1))
        y = F.relu(y)
        combined = torch.cat((x, y), 1)
        return F.relu(self.fc1(combined))

# Policy

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv_grid = ConvGridNet() 
        self.out = nn.Linear(512, 4 + 1, bias=False) # For both action and expected value

    def forward(self, input_grid, input_player):
        x = self.conv_grid(input_grid, input_player)
        x = self.out(x)
        # Split last five outputs into scores and value
        scores = x[:,:4]
        value = x[:,4]
        return scores, value

# Selecting actions


def select_action(e, state):
    drop = interpolate(e, DROP_MAX, DROP_MIN, DROP_OVER)
    
    state_grid = Variable(torch.from_numpy(state[0]).float())
    state_player = Variable(torch.from_numpy(state[1]).float())
    scores, value = policy(state_grid, state_player) # Forward state through network
    scores = F.dropout(scores, drop, True) # Dropout for exploration
    scores = F.softmax(scores)
    #print('scores: ', scores)
    action = scores.multinomial() # Sample an action
    return action, value

def finish_episode(actions, values, rewards):
    global optimizer
    # Calculate discounted rewards, going backwards from end
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.Tensor(discounted_rewards)

    # Use REINFORCE on chosen actions and associated discounted rewards
    value_loss = 0
    count = 0
    for action, value, reward in zip(actions, values, discounted_rewards):
        count += 1
        reward_diff = reward - value.data[0] # Treat critic value as baseline
        action.reinforce(reward_diff) # Try to perform better than baseline
        value_loss += mse(value, Variable(torch.Tensor([reward]))) # Compare with actual reward
    # Backpropagate
    optimizer.zero_grad()
    nodes = [value_loss] + actions
    gradients = [torch.ones(1)] + [None for _ in actions] # No gradients for reinforced values
    autograd.backward(nodes, gradients)
    optimizer.step()
    
    return discounted_rewards, value_loss
    
###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    global log_every, render_every, count_episodes, env, policy, optimizer, reward_avg, value_avg, scores_avg
    global action, actions, value, values, rewards, done, state, states, score, scores, firstAction
    hidden_size = 200
    learning_rate = 1e-4
    weight_decay = 1e-5

    log_every = 1000
    render_every = 20000

    # Create environment
    env = Environment(mazeWidth, mazeHeight)
    state = env.reset(mazeMap, playerLocation, opponentLocation, piecesOfCheese)
    # Create policy
    
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)#, weight_decay=weight_decay)
    count_episodes = 0
    if os.path.isfile(file_name):
        print("=> loading checkpoint =>")
        checkpoint = torch.load(file_name)
        policy.load_state_dict(checkpoint['state_dict'])
        count_episodes = checkpoint['count_episodes']
        reward_avg = checkpoint['reward_avg']
        scores_avg = checkpoint['scores_avg']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found =>")
        if os.path.isfile('log.txt'):
            os.remove('log.txt')        
        reward_avg = SlidingAverage('reward_avg', steps=log_every)
        scores_avg = SlidingAverage('score_avg', steps=log_every)

    value_avg = SlidingAverage('value avg', steps=log_every)

    
    actions = []
    values = []
    rewards = []
    states = []
    scores = []
    done = False
    firstAction = True
    
    
###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move

def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):   
    global log_every, render_every, count_episodes, env, policy, optimizer, reward_avg, value_avg, scores_avg
    global action, actions, value, values, rewards, done, state, states, score, scores, firstAction
    
    # Receives the reward of the last action 
    next_state, reward, action, score = env.step(mazeMap, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, False)

    if not firstAction:
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        states.append(state)
        scores.append(score)
        reward_avg.add(reward)
        scores_avg.add(score)
        if count_episodes % log_every == 0:
            f = open('log.txt','a',newline='')
            writer=csv.writer(f)
            writer.writerow([count_episodes, reward_avg, scores_avg])
            print('[episode=%d]' % count_episodes, reward_avg, scores_avg)
            f.close()
    firstAction = False
    output, value = select_action(count_episodes, next_state)
    state = next_state
    count_episodes += 1
    env.update_action(output)
    return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][int(output.data.numpy())]

###############################
# Postprocessing function
# The postprocessing function is called at the end of a game
# It can be used to perform intensive computations that can be
# used for upcoming games.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything

def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    global log_every, render_every, count_episodes, env, policy, optimizer, reward_avg, value_avg, action
    global actions, values, rewards, done, state, states

    next_state, reward, action, score = env.step(mazeMap, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, True)
    actions.append(action)
    rewards.append(reward)
    values.append(value)
    states.append(state)
    scores.append(score)
    reward_avg.add(reward)
    scores_avg.add(score)
  
    discounted_rewards, value_loss = finish_episode(actions, values, rewards)
    count_episodes += 1
    save_checkpoint({'state_dict': policy.state_dict(), 'count_episodes':count_episodes, 'reward_avg':reward_avg, 'scores_avg':scores_avg, 'optimizer' : optimizer.state_dict()})
    return
    
