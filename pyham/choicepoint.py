from typing import List, NamedTuple
from gym import spaces
import numpy as np


class Choicepoint(NamedTuple):
  name: str
  choice_space: spaces.Space
  discount: float
  id: int=-1

class ChoicepointsManager:
  def __init__(self, eval=False):
    """
    Parameters:
      choicepoints
      eval: if `True`, ignore all choicepoint's discount.
    Attrubutes:
      choicepoints
      choicepoints_order
      N
    """
    self.N = 0
    self.choicepoints_order = []
    self.tau = []
    self.choicepoints = {}
    self.eval = eval

  def set_eval(self, eval):
    if eval:
      self.init_discounts = np.ones((self.N,))
    else:
      self.init_discounts = np.array([
        self.choicepoints[name].discount 
        for name in self.choicepoints_order
      ])

  def add_choicepoint(self, choicepoint: Choicepoint):
    """
    """
    assert(not choicepoint.name in self.choicepoints_order)
    choicepoint = choicepoint._replace(id=self.N)
    self.N+=1
    self.choicepoints_order.append(choicepoint.name)
    self.choicepoints[choicepoint.name] = choicepoint

    # Reward calculation setup
    if self.eval:
      self.init_discounts = np.ones((self.N,))
    else:
      self.init_discounts = np.array([
        self.choicepoints[name].discount 
        for name in self.choicepoints_order
      ])

  def reset(self):
    """
      Episodic reset
    """
    self.cumulative_rewards = np.zeros((self.N,))
    self.discounts = np.ones((self.N,))
    self.tau = np.zeros((self.N,))

  def distribute_reward(self, reward: float):
    if self.N==0:
      return 0
    rewards = self.discounts*reward
    self.cumulative_rewards+=rewards
    self.discounts*=self.init_discounts
    self.tau+=1

  def reset_choicepoint(self, cp_name: str):
    cp_idx = self.choicepoints[cp_name].id
    
    cp_cumulative_reward = self.cumulative_rewards[cp_idx]
    cp_tau = self.tau[cp_idx]

    self.cumulative_rewards[cp_idx]=0
    self.discounts[cp_idx]=1
    self.tau[cp_idx]=0

    return cp_cumulative_reward, cp_tau
  
  def __len__(self):
    return self.N

  def __getitem__(self, index):
    return self.choicepoints[self.choicepoints_order[index]]