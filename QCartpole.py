import gym
import random
import numpy as np
import time
import math
from collections import deque
import plotly
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='vtomar', api_key='DWGsIthCdaKv9M5fpylN')

#import matplotlib.pyplot as plt
#from ggplot import *

#env.observation_space: cart position(x), cart velocity(x'), angle (theta), angle velocity (theta')
#env.observation_space.high: array([4.80000000e+00, 3.40282347e+38, 4.18879020e-01, 3.40282347e+38])
#env.observation_space.low: array([-4.80000000e+00, -3.40282347e+38, -4.18879020e-01, -3.40282347e+38])
'''
cart position range [-4.8, 4.8]
cart velocity range [-3.40282347e+38, 3.40282347e+38] large velocity domains
angle (theta) range: [-4.18879020e-01, 4.18879020e-01]
angle velocity (theta) range: [-3.40282347e+38, 3.40282347e+38] large velocity domains
'''

#converting continuous space to discrete action_space
#to reduce dimensionality we do not need cart position and velocity since we only have 200 timesteps
#Scaling theta (angle) down to disrete interval theta [0, 1, 2, 3, 4, 5, 6]
#Scaling theta' (angle velocity) down to discrete interval [0-12] (integers only)

buckets = (1,1,6,12) # down scaling feature space to discrete range
numEpisodes = 1000 # training episodes
nWinTicks = 195 # average ticks over 100 episodes required for win
minLearningRate_Alpha = 0.1
minExplorationRate_epsilon = 0.1
discountFactor_gamma = 1
adaDivisor = 25
quiet = False
monitor = False
maxEnvSteps = None

env = gym.make("CartPole-v0")

if maxEnvSteps is not None:
    env._max_episode_steps = maxEnvSteps

if monitor:
    env = gym.wrappers.Monitor(env, 'tmp/cartpole-1', force = True) # record results for upload

qTable = np.zeros(buckets + (env.action_space.n,))

#Discretize state space
upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]

scores = deque(maxlen=100)

scoreList = []
explorationRates = []

meanScore = 0
for episode in range(numEpisodes):

    obs = env.reset()
    #discretize the state
    #for i in range(len(obs)):
    #    print("Obs:{} LowerBound:{} UpperBound:{}".format(obs[i], lower_bounds[i], upper_bounds[i]))
    ratios = [(obs[i] + abs(lower_bounds[i]))/(upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    #ratios = [obs[i]/(upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    #print("Ratios:{}".format(ratios))
    currentState = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    #print("CurrentState:{}".format(currentState))
    currentState = [min(buckets[i] - 1, max(0, currentState[i])) for i in range(len(obs))]
    #print("Updated CurrentState:{}".format(currentState))
    currentState = tuple(currentState)


    learningRate_Alpha = max(minLearningRate_Alpha, min(1.0, 1.0 - math.log10((episode+1)/adaDivisor)))
    explorationRate_epsilon = max(minExplorationRate_epsilon, min(1.0, 1.0 - math.log10((episode+1)/adaDivisor)))
    #explorationRate_epsilon = max(minExplorationRate_epsilon, 1/(episode+1))
    explorationRates.append(explorationRate_epsilon)
    done = False

    i = 0

    while not done:
        env.render()
        action = env.action_space.sample() if (np.random.random() <= explorationRate_epsilon) else np.argmax(qTable[currentState])
        obs, reward, done, _ = env.step(action)

        #discretize the state
        ratios = [(obs[i] + abs(lower_bounds[i]))/(upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        #print("Ratios:{}".format(ratios))
        newState = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        #print("NewState:{}".format(newState))
        newState = [min(buckets[i] - 1, max(0, newState[i])) for i in range(len(obs))]
        #print("Updated NewState:{}".format(newState))
        newState = tuple(newState)
        #print("{} {}".format(learningRate_Alpha, discountFactor_gamma))
        qTable[currentState][action] += learningRate_Alpha * (reward + discountFactor_gamma * np.max(qTable[newState]) - qTable[currentState][action])
        currentState = newState
        i+=1
    scores.append(i)
    scoreList.append(i)
    print("Episode:{} Lasted {} observations".format(episode, i))
    meanScore = np.mean(scores)
    #print("MeanScore:{}".format(meanScore))
    if meanScore >= nWinTicks and episode >= 100:
        if not quiet:
            print('Ran {} episodes. Solved after {} trials ✔'.format(episode, episode - 100))
        break
    if episode % 100 == 0 and not quiet:
        print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(episode, meanScore))

print("Final Mean Score:{}".format(meanScore))
env.close()

trace0 = go.Scatter(x = list(range(len(scoreList))), y = scoreList, mode = 'lines')
trace1 = go.Scatter(x = list(range(len(scoreList))), y = explorationRates, mode = 'lines')

fig = tools.make_subplots(rows = 2, cols = 1)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)

py.iplot(fig, filename= 'simple-subplot-with-annotations')


#data0 = [trace0]
#py.iplot(data0, filename = 'basic-scatter')

#ggplot() + geom_
#exec(open("QCartpole.py").read())
