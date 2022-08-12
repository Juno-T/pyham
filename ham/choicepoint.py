from typing import List, NamedTuple
from gym import spaces
import numpy as np

from pyham.ham import HAM

class Choicepoint(NamedTuple):
  name: str
  choice_space: spaces.Space
  discount: float
  _id: int=-1

class ChoicepointsManager:
  def __init__(self, choicepoints: List[Choicepoint], eval=False):
    """
    Parameters:
      choicepoints
      eval: if `True`, ignore all choicepoint's discount.
    Attrubutes:
      choicepoints
      choicepoints_order
      N
    """
    self.choicepoints = {}
    self.choicepoints_order = []
    self.N = 0
    self.choicepoints = {
      cp.name: cp
      for cp in choicepoints
    }
    self.choicepoints_order = list(self.choicepoints.keys())
    for i, name in enumerate(self.choicepoints_order):
      self.choicepoints[name]._id = i

    self.N = len(self.choicepoints_order)
    self.cumulative_rewards = np.zeros((self.N,))
    if eval:
      self.init_discounts = np.ones((self.N,))
    else:
      self.init_discounts = np.array([
        self.choicepoints[name].discount 
        for name in self.choicepoints_order
      ])

  def reset(self):
    self.discounts = np.ones((self.N,))
    self.cumulative_rewards = np.zeros((self.N,))

  def distribute_reward(self, reward: float):
    rewards = self.discounts*reward
    self.cumulative_rewards+=rewards
    self.discounts*=self.init_discounts

  def reset_choicepoint(self, cp_name: str):
    cp_idx = self.choicepoints[cp_name]._id
    
    cp_cumulative_reward = self.cumulative_rewards[cp_idx]

    self.cumulative_reward[cp_idx]=0
    self.discounts[cp_idx]=1

    return cp_cumulative_reward
  