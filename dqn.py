# -*- coding: utf-8 -*-

"""

Episode 100  AvgScore: -177.38  Eps: 0.6058  AvgTime: 0.5s/ep  Land/Crash:  0/100
Episode 200  AvgScore: -113.34  Eps: 0.3670  AvgTime: 1.3s/ep  Land/Crash:  9/91
Episode 300  AvgScore:  -58.16  Eps: 0.2223  AvgTime: 4.2s/ep  Land/Crash: 68/32
Episode 400  AvgScore:    9.51  Eps: 0.1347  AvgTime: 4.9s/ep  Land/Crash: 95/ 5
Episode 500  AvgScore:   36.44  Eps: 0.0816  AvgTime: 4.9s/ep  Land/Crash: 95/ 5
Episode 600  AvgScore:  130.75  Eps: 0.0494  AvgTime: 3.6s/ep  Land/Crash: 85/15
Episode 700  AvgScore:  197.99  Eps: 0.0299  AvgTime: 2.3s/ep  Land/Crash: 92/ 8
Episode 800  AvgScore:  199.60  Eps: 0.0181  AvgTime: 2.0s/ep  Land/Crash: 87/13
Episode 801  AvgScore:  200.04  Eps: 0.0180  Time: 2.0s /  424 steps  LANDED
Environment solved in 801 episodes!     Average Score: 200.04


"""

import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time

import torch


env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


from dqn_agent import Agent, get_device_info

def dqn_train(agent, n_episodes=2000, max_t=1000, 
              eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    t_last = time()
    _C = 0
    _L = 0 
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        t_start = time()
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        if reward == -100:
          _end = 'CRASH '
          _C += 1
        else:
          _end = 'LANDED'
          _L += 1
        t_ep = time() - t_start
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {:>4}  AvgScore: {:>7.2f}  Eps: {:.4f}  Time: {:.1f}s / {:>4} steps  {}'.format(
            i_episode, np.mean(scores_window), eps, t_ep, t, _end), end="", flush=True)
        if i_episode % 100 == 0:
            t_lap = time() - t_last
            t_last = time()
            print('\rEpisode {:>4}  AvgScore: {:>7.2f}  Eps: {:.4f}  AvgTime: {:.1f}s/ep  Land/Crash: {:>2}/{:>2}'.format(
                i_episode, np.mean(scores_window), eps, t_lap / 100, _L, _C), flush=True)
            _L = 0
            _C = 0
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


n, m, p = get_device_info()
if n:
  print("GPU: {}, Mem:{:.1f}GB, Procs:{}".format(n, m, p))
  
settings = [
    {"double" : True,
     "dueling" : True}
    ]

for setting in settings:
  
  double_dqn = setting['double']
  dueling = setting['dueling']
  
  eng = Agent(state_size=8, action_size=4, seed=0, 
              double_dqn=double_dqn, dueling=dueling)  
  
  scores = dqn_train(agent=eng)
  
  # plot the scores
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(np.arange(len(scores)), scores)
  plt.ylabel('Score')
  plt.xlabel('Episode #')
  plt.show()
