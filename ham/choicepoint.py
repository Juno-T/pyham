from typing import List, NamedTuple
from gym import spaces
import numpy as np

from pyham.ham import HAM

class Choicepoint(NamedTuple):
  name: str
  choice_space: spaces.Space
  discount: float
  id: int=-1

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
    self.N = len(choicepoints)
    self.choicepoints_order = [cp.name for cp in choicepoints]
    self.choicepoints = {
      cp.name: cp._replace(id=i)
      for i, cp in enumerate(choicepoints)
    }

    # Reward calculation setup
    if eval:
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
    self.discounts = np.ones((self.N,))
    self.cumulative_rewards = np.zeros((self.N,))

  def distribute_reward(self, reward: float):
    rewards = self.discounts*reward
    self.cumulative_rewards+=rewards
    self.discounts*=self.init_discounts

  def reset_choicepoint(self, cp_name: str):
    cp_idx = self.choicepoints[cp_name].id
    
    cp_cumulative_reward = self.cumulative_rewards[cp_idx]

    self.cumulative_rewards[cp_idx]=0
    self.discounts[cp_idx]=1

    return cp_cumulative_reward
  