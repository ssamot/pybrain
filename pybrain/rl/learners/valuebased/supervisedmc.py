__author__ = 'Spyridon Samothrakis, spyridon.samothrakis@gmail.com'

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.structure.modules.module import Module
import numpy as np
from random import randint
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer

class SupervisedMC(ValueBasedLearner):
    """ State-Action-Reward-State-Action (SARSA) algorithm.
        SARSA(1), i.e. MC-Control using Supervised Learning
        Works best when the MDP is a tree, not a graph (i.e. has absorbing states and no state can be visited twice)
    """


    def __init__(self, numActions, lower_bounds, upper_bounds, 
                 force_absorbing = True ,  
                 horizon=5000, gamma=0.99, prior = 0, n_samples = 3000):
        ValueBasedLearner.__init__(self)
        self.explorer = EpsilonGreedyExplorer(0.4,0.9999)
        self.force_absorbing = force_absorbing
        self.horizon = horizon
        self.gamma = gamma
        self.n_samples = n_samples
        self.laststate = None
        self.lastaction = None
        assert(len(lower_bounds) == len(upper_bounds))        
        
        X = []
        for i in range(n_samples):
            sample = [0]*len(lower_bounds)
            for b in range(len(upper_bounds)):
                lb = lower_bounds[b]
                ub = upper_bounds[b]
                sample[b] = randint(lb,ub)
            sample.append(randint(0,numActions-1))
            #print sample
            X.append(sample)
        self.prior_X = np.array(X)                
        self.prior_y = np.zeros(self.prior_X.shape[0])*prior
        
                
    def learn(self):
        #print self.dataset
        samples = self.dataset
        
        clf = self.module.clf
        # Convert to np.arrays
        X = []
        y = []
        total_reward = 0
        for seq in samples:
            prev_reward = -10
            for i, (state, action, reward) in enumerate(seq):
                #print state, action, reward
                example = np.append(state, action)
                if(self.force_absorbing):
                    if(prev_reward > reward[0] ):
                        break
                #print action[0]
                X.append(example)
                #print i, np.power(self.gamma,i), reward[0]
                total_reward+=np.power(self.gamma,i)*reward[0]
                prev_reward = reward[0]
       
        #exit(0)
        X = np.array(X)                
        y = np.ones(X.shape[0])*total_reward
        #if(self.module.initialised):
        #    y_old = clf.predict(self.X)
        #    y = 0.01*y_old + y
        if hasattr(self, 'X'):
            self.X = np.append(self.X,X, axis = 0)
            self.y = np.append(self.y,y)
        else:
            self.X = X
            self.y = y
            
        
        self.X = self.X[-self.horizon:]
        self.y = self.y[-self.horizon:]
        
        X = np.append(self.X,self.prior_X, axis = 0)
        y = np.append(self.y,self.prior_y)
        #del self.X[1]
        # print X
        #print total_reward
        #print self.X.shape,self.y.shape
        print total_reward, self.explorer.epsilon, X.shape[0]
        clf.fit(X,y)
        self.module.initialised = True
                


class CLFModuleWrapper(Module):
    """ Wrapper Class so that sci-kit learn classifiers can be used 
        in pybrain
    """

    def __init__(self, clf, feature_length,numActions, name = None):
        """ initialize with the number of rows and columns. the table
            values are all set to zero.
        """
        Module.__init__(self, feature_length, 1, name)
        self.numActions = numActions
        self.clf = clf
        self.initialised = False
       

    def _forwardImplementation(self, inbuf, outbuf):
        """ takes two coordinates, row and column, and returns the
            value in the table.
        """
        if(not self.initialised):
            #print type(inbuf)
            outbuf[0] = randint(0,self.numActions-1)
        else:
            values = [self.clf.predict([np.append(inbuf,i)])[0] for i  in xrange(0,self.numActions)]
            #values += np.random.rand(self.numActions)*0.01
            outbuf[0] = np.argmax(values)
            if((values==values[0]).all()):
                #print values
                outbuf[0] = randint(0,self.numActions-1)
            #print inbuf, outbuf
           

   

