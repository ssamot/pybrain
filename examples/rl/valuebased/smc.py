#!/usr/bin/env python
__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

""" This example demonstrates how to use the discrete Temporal Difference
Reinforcement Learning algorithms (SARSA, Q, Q(lambda)) in a classical
fully observable MDP maze task. The goal point is the top right free
field. """

from scipy import * #@UnusedWildImport
import pylab

from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.learners.valuebased.supervisedmc import SupervisedMC, CLFModuleWrapper
from sklearn.tree import DecisionTreeRegressor as clf
from pybrain.rl.agents import LearningAgent
import numpy as np
from pybrain.rl.experiments import Experiment


# create the maze with walls (1)
envmatrix = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0, 0, 1, 0, 1],
                   [1, 0, 0, 1, 0, 0, 1, 0, 1],
                   [1, 0, 0, 1, 0, 1, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 1, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

env = Maze(envmatrix, (7, 7))


# create task
task = MDPMazeTask(env)
n_actions = 4
learner = SupervisedMC(n_actions, [0], [80])
cmw = CLFModuleWrapper(clf(min_samples_split = 1),1,n_actions,"DecisionTreeRegressor")
# standard exploration is e-greedy, but a different type can be chosen as well
# learner.explorer = BoltzmannExplorer()

# create agent
agent = LearningAgent(cmw, learner)




# create experiment
experiment = Experiment(task, agent)

# prepare plotting
pylab.gray()
pylab.ion()

for i in range(1000):

    # interact with the environment (here in batch mode)
    experiment.doInteractions(100)
    agent.learn()
    agent.reset()

    # and draw the table
    values = [learner.module.clf.predict([i,j])[0] for i in xrange(0,81) for j in xrange(n_actions)]
    values = np.array(values)
    values = values.reshape(81,4).max(1).reshape(9,9)
   
    values[7,7] = values.max() +0.1
    #values[values >= 1] = 1
    pylab.pcolormesh(values,edgecolor = "red")
    
#    for i,val in enumerate(values):
#        print i,val
#    print "*"*80
    #print envmatrix
    #pylab.pcolor(envmatrix, edgecolor = "red")
    pylab.draw()
