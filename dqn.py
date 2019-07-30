# -*- coding: utf-8 -*-

"""
DD-DQN:
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



   Solved  NrEpisode  Time(s)  double  dueling       PER  eps_start  eps_min  eps_decay  opt      lr
2    True        684   2525.9    True     True  tree_per       1.00     0.01      0.995  sgd  0.0500

0   False       2000   8390.6    True     True  tree_per       0.01     0.01      1.000  sgd  0.0500
1   False       2000   5781.0    True     True  tree_per       0.01     0.01      1.000  sgd  0.0005
3   False       2000   9193.6    True     True  tree_per       1.00     0.01      0.995  sgd  0.0005

"""

import gym
import numpy as np
from collections import deque, OrderedDict
import matplotlib.pyplot as plt
from time import time

import torch



from dqn_agent import Agent

def dqn_train(agent, env, n_episodes=2000, max_t=1000, 
              ):
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
  solved = False
  for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    t_start = time()
    for t in range(max_t):
        action = agent.act(state, use_eps=True)
        next_state, reward, done, _ = env.step(action)
        agent.episode = i_episode
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
    eps = agent.eps_step() # decrease epsilon
    print('\rEp {:>4}  AvgScr: {:>7.2f}  Eps: {:.4f}  Time: {:.1f}s / {:>4} steps  {} @ {}'.format(
        i_episode, np.mean(scores_window), eps, t_ep, t, _end, score), end="", flush=True)
    if i_episode % 100 == 0:
      t_lap = time() - t_last
      t_last = time()
      print('\rEpisode {:>4}  AvgScore: {:>7.2f}  Eps: {:.4f}  AvgTime: {:.1f}s/ep  Land/Crash: {:>2}/{:>2}'.format(
          i_episode, np.mean(scores_window), eps, t_lap / 100, _L, _C), flush=True)
      _L = 0
      _C = 0
    if np.mean(scores_window)>=200.0:
      print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
      torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
      solved = True
      break     
       
  return scores, solved



  


if __name__ == '__main__':
  
  env = gym.make('LunarLander-v2')
  env.seed(0)
  obs_shape = env.observation_space.shape
  n_act = env.action_space.n
  print('State shape: ', obs_shape)
  print('Number of actions: ', n_act)
  
  settings = [
#      {
#        "double"    : True,
#        "dueling"   : True,
#        "PER"       : "tree_per",
#        "eps_start" : 0.01,
#        "eps_min"   : 0.01,
#        "eps_decay" : 1.,
#        "opt"       : "sgd",
#        "lr"        : 5e-02,
#       },
#
#      {
#        "double"    : True,
#        "dueling"   : True,
#        "PER"       : "tree_per",
#        "eps_start" : 0.01,
#        "eps_min"   : 0.01,
#        "eps_decay" : 1.,
#        "opt"       : "sgd",
#        "lr"        : 5e-04,
#       },
#
#      {
#        "double"    : True,
#        "dueling"   : True,
#        "PER"       : "tree_per",
#        "eps_start" : 1.0,
#        "eps_min"   : 0.01,
#        "eps_decay" : 0.995,
#        "opt"       : "sgd",
#        "lr"        : 5e-2,
#       },
#
#      {
#        "double"    : True,
#        "dueling"   : True,
#        "PER"       : "tree_per",
#        "eps_start" : 1.0,
#        "eps_min"   : 0.01,
#        "eps_decay" : 0.995,
#        "opt"       : "sgd",
#        "lr"        : 5e-4,
#       },

      {
        "double"    : True,
        "dueling"   : True,
        "PER"       : "tree_per",
        "eps_start" : 0.01,
        "eps_min"   : 0.01,
        "eps_decay" : 1.,
        "opt"       : "adam",
        "lr"        : 5e-5,
       },

      {
        "double"    : True,
        "dueling"   : True,
        "PER"       : "tree_per",
        "eps_start" : 1.0,
        "eps_min"   : 0.01,
        "eps_decay" : 0.995,
        "opt"       : "adam",
        "lr"        : 5e-5,
       },
  
      {
        "double"    : True,
        "dueling"   : True,
        "PER"       : "naive_per",
        "eps_start" : 1.0,
        "eps_min"   : 0.01,
        "eps_decay" : 0.995,
        "opt"       : "sgd",
        "lr"        : 5e-2,
      },
  
      {
        "double"    : True,
        "dueling"   : True,
        "PER"       : "naive_per",
        "eps_start" : 0.01,
        "eps_min"   : 0.01,
        "eps_decay" : 1.,
        "opt"       : "sgd",
        "lr"        : 5e-2,
      },

      {
        "double"    : True,
        "dueling"   : True,
        "PER"       : "naive_per",
        "eps_start" : 1.0,
        "eps_min"   : 0.01,
        "eps_decay" : 0.995,
        "opt"       : "adam",
        "lr"        : 5e-5,
      },
  
      {
        "double"    : True,
        "dueling"   : True,
        "PER"       : "naive_per",
        "eps_start" : 0.01,
        "eps_min"   : 0.01,
        "eps_decay" : 1.,
        "opt"       : "adam",
        "lr"        : 5e-5,
      },
  
      {
        "double"    : True,
        "dueling"   : True,
        "PER"       : None,
        "eps_start" : 1.0,
        "eps_min"   : 0.01,
        "eps_decay" : 0.995,        
        "opt"       : "adam",
        "lr"        : 5e-4,
      },       
    ]
  
  dct_res = OrderedDict({
      "Solved" : [],
      "NrEpisode" : [],
      "Time(s)": []
      })
  for k in settings[0]:
    dct_res[k] = []
    
  eps_start = 1.0
  eps_min = 0.01
  eps_decay = 0.995
  
  for setting in settings:
    
    _s1 = "Grid searching "
    _s2 = ""
    for k in setting:
      dct_res[k].append(setting[k])
      _s2 += "[{} {}]".format(k+":", setting[k])
    
    print("\n"+_s1+_s2)
    
    eng = Agent(state_size=obs_shape[0], action_size=n_act, seed=0, 
                **setting)  
    
    eps_start = 1.0 #0.4 if PER is not None else 1.0
    t_start = time()
    scores, solved = dqn_train(agent=eng, env=env)
    t_end = time()
    
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.title(_s2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    dct_res['Solved'].append(solved)
    dct_res['NrEpisode'].append(len(scores))
    dct_res['Time(s)'].append(round(t_end - t_start, 1))
  
  import pandas as pd
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)  
  df = pd.DataFrame(dct_res)
  print(df.sort_values('NrEpisode'))
