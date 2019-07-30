import numpy as np
import random

from model import QNetwork
from per import SimpleReplayBuffer, PERMemory, NaivePrioritizedBuffer
import torch as th
import torch.optim as optim
import torch.nn as nn



def get_device_info(dev=None):  
  if th.cuda.is_available():
    if dev is None:
      dev = th.device("cuda:0")
    if int(th.__version__[0]) > 0:
      _dev = dev
    else:
      _dev = dev.index
    gpu = th.cuda.get_device_properties(_dev)
    return gpu.name, gpu.total_memory / (1024**3), gpu.multi_processor_count
  else:
    return None, None, None

class Agent():
  """Interacts with and learns from the environment."""

  
  def __init__(self, state_size, action_size, seed, 
               double=False, dueling=False, PER=None,
               opt='sgd', lr=5e-5,
               eps_start=0.01, eps_decay=1., eps_min=0.01, 
               GAMMA=0.99, TAU=1e-3, BATCH_SIZE=32, BUFFER_SIZE=int(1e6), UPDATE_EVERY=4,
               FULL_DEBUG=False):
    """Initialize an Agent object.
    
    Params
    ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        seed (int): random seed
        double_dqn: use DDQN
        dueling: use adv/value stream split in model
    """
    self.GAMMA = GAMMA
    self.TAU = TAU
    self.BATCH_SIZE = BATCH_SIZE
    self.BUFFER_SIZE = BUFFER_SIZE
    self.UPDATE_EVERY = UPDATE_EVERY
    self.episode = -1
    self.eps = eps_start
    self.eps_decay = eps_decay
    self.eps_min = eps_min
    self.state_size = state_size
    self.action_size = action_size
    self.double_dqn = double
    self.dueling = dueling
    self.seed = random.seed(seed)
    self.FULL_DEBUG = FULL_DEBUG
    np.random.seed(seed)
    
    self.PER = bool(PER)
    
    self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    
    # Q-Network
    self.qnetwork_local = QNetwork(state_size, action_size, seed,
                                   dueling=self.dueling).to(self.device)
    self.qnetwork_target = QNetwork(state_size, action_size, seed,
                                    dueling=self.dueling).to(self.device)
    if opt.lower() == 'adam':
      self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
    elif opt.lower() == 'sgd':
      self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=lr)
    else:
      raise ValueError("Uknown optimizer ''".format(opt))
      
    self.loss_func = nn.MSELoss()
    
    self.hard_update(self.qnetwork_local, self.qnetwork_target)

    # Replay memory
    if self.PER:
      if "tree" in PER:
        self.memory = PERMemory(capacity=BUFFER_SIZE, 
                                engine='torch', device=self.device)
      elif "naive" in PER:
        self.memory = NaivePrioritizedBuffer(capacity=BUFFER_SIZE,
                                             engine='torch', device=self.device)
      else:
        raise ValueError("Uknown memory engine")        
    else:
      self.memory = SimpleReplayBuffer(capacity=BUFFER_SIZE, seed=seed,
                                       engine='torch', device=self.device)
      
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
    if self.FULL_DEBUG:
      print("Agent model:")
      print(self.qnetwork_local)
    var_device = next(self.qnetwork_local.denses.parameters()).device
    dev_name, dev_mem, dev_proc = get_device_info(var_device)
    print(" Running on '{}':{}  Mem:{}  Procs: {}".format(
        var_device, dev_name, dev_mem, dev_proc))
    print(" Double DQN:  {}".format(self.double_dqn))
    print(" Dueling DQN: {}".format(self.dueling))
    print(" Optimizer:   {}".format(self.optimizer.__class__))
    print(" Using PER:   {} [{} | {}]\n".format(self.PER, PER, self.memory.__class__))
    return
    
    
  def eps_step(self):
    self.eps = max(self.eps_min, self.eps_decay * self.eps)
    return self.eps
  
  def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, next_state, done)
    
    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
    if self.t_step == 0:
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.BATCH_SIZE:
            self.learn(self.GAMMA)

  def act(self, state, use_eps=False):
    """Returns actions for given state as per current policy.
    
    Params
    ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
    """
    state = th.from_numpy(state).float().unsqueeze(0).to(self.device)
    self.qnetwork_local.eval()  # set infer mode
    with th.no_grad():
        action_values = self.qnetwork_local(state)
    self.qnetwork_local.train() # go back to train mode

    # Epsilon-greedy action selection
    if use_eps and (random.random() < self.eps):
      return random.choice(np.arange(self.action_size))
    else:
      return np.argmax(action_values.cpu().data.numpy())

  def learn(self, gamma):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
        experiences (Tuple[th.Variable]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
    """
    self.memory.episode = self.episode
    if self.PER:
      experiences, tree_idxs, IS_weights = self.memory.sample(self.BATCH_SIZE)
    else:
      experiences = self.memory.sample(self.BATCH_SIZE)
      
    states, actions, rewards, next_states, dones = experiences                
            
    target_values = self.qnetwork_target(next_states).detach()
    
    if self.double_dqn:
        local_next_values = self.qnetwork_local(next_states)
        _, local_next_actions = th.max(local_next_values, 1)
        local_next_actions = local_next_actions.detach().unsqueeze(1)
        next_max_values = th.gather(target_values, 1, local_next_actions)
    else:
      next_max_values, next_best_actions = target_values.max(1)
      next_max_values = next_max_values.unsqueeze(1)
      
    targets = rewards + gamma * next_max_values * (1 - dones)
    
    outputs = self.qnetwork_local(states)
    
    selected_outputs = th.gather(outputs, 1, actions)
    
    th_residual = selected_outputs - targets
    
    if th.isnan(th_residual).any():
      raise ValueError("Residual tensor contains nans!")
    
    if self.PER:
      np_errors = th.abs(th_residual).cpu().detach().numpy()
      self.memory.batch_update(tree_idxs, np_errors)
    
    self.optimizer.zero_grad()
    
    if self.PER:
      loss_batch = th_residual.pow(2) * IS_weights
      loss = loss_batch.mean()
    else:
      loss = self.loss_func(selected_outputs, targets)      
    loss.backward()
    th.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)                     

  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


  def hard_update(self, local_model, target_model):
      target_model.load_state_dict(local_model.state_dict())
