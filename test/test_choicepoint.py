import unittest
import pytest
from gym import spaces
import numpy as np
from numpy.random import default_rng

from pyham.ham import Choicepoint, ChoicepointsManager


class TestFunctionality(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    self.cp_list=[
      Choicepoint("cp1", spaces.Discrete(2), 1.0),
      Choicepoint("cp2", spaces.Discrete(3), 0.5),
      Choicepoint("cp3", spaces.Discrete(4), 0.8),
    ]
    self.cpm = ChoicepointsManager(eval=False)
    for cp in self.cp_list:
      self.cpm.add_choicepoint(cp)
    return super().setUp()

  def test_initialize(self):
    # use manual init cpm
    cpm = ChoicepointsManager(eval=False)
    for cp in self.cp_list:
      cpm.add_choicepoint(cp)
    for cp in self.cp_list:
      self.assertTrue(cp.name in cpm.choicepoints.keys())
      self.assertTrue(cp.name in cpm.choicepoints_order)
    for i, cp_name in enumerate(cpm.choicepoints_order):
      self.assertTrue(cpm.choicepoints[cp_name].id==i)
    
    self.assertEqual(cpm.N, len(self.cp_list))
    self.assertTrue(np.sum(cpm.init_discounts)==sum([cp.discount for cp in self.cp_list]))

  def test_reset(self):
    self.cpm.reset()
    self.assertTrue(np.array_equal(self.cpm.discounts, np.ones(len(self.cp_list))))
    self.assertTrue(np.array_equal(self.cpm.cumulative_rewards, np.zeros(len(self.cp_list))))

  def test_distribute_reward(self):
    # use default cpm
    self.cpm.reset()
    cumulative_rewards = np.zeros(len(self.cp_list))

    # reward=1
    self.cpm.distribute_reward(1)
    cumulative_rewards+=np.array([1,1,1])
    self.assertTrue(np.array_equal(self.cpm.cumulative_rewards, cumulative_rewards))
    self.assertTrue(np.array_equal(self.cpm.discounts, np.array([1,0.5,0.8])))
    self.assertTrue(np.array_equal(self.cpm.init_discounts, np.array([1,0.5,0.8])))
    self.assertTrue(np.array_equal(self.cpm.tau, np.array([1,1,1])))

    # reward=2
    self.cpm.distribute_reward(2)
    cumulative_rewards+=np.array([2*1,2*0.5,2*0.8])
    self.assertTrue(np.array_equal(self.cpm.cumulative_rewards, cumulative_rewards))
    self.assertTrue(np.array_equal(self.cpm.discounts, np.array([1,0.5**2,0.8**2])))
    self.assertTrue(np.array_equal(self.cpm.init_discounts, np.array([1,0.5,0.8])))
    self.assertTrue(np.array_equal(self.cpm.tau, np.array([1,1,1])*2))

    # reward=-1
    self.cpm.distribute_reward(-1)
    cumulative_rewards+=np.array([-1*1,-1*0.5**2,-1*0.8**2])
    self.assertTrue(np.array_equal(self.cpm.cumulative_rewards, cumulative_rewards))
    self.assertTrue(np.array_equal(self.cpm.discounts, np.array([1,0.5**3,0.8**3])))
    self.assertTrue(np.array_equal(self.cpm.init_discounts, np.array([1,0.5,0.8])))
    self.assertTrue(np.array_equal(self.cpm.tau, np.array([1,1,1])*3))

  def test_choicepoint(self):
    # use default cpm
    self.cpm.reset()

    self.cpm.distribute_reward(1)
    self.cpm.distribute_reward(1)
    cp1_rew, cp1_tau = self.cpm.reset_choicepoint("cp1")
    self.assertTrue(cp1_rew==1+1*1.0)
    self.assertTrue(cp1_tau==2)
    self.assertTrue(np.array_equal(self.cpm.discounts, np.array([1.0, 0.5**2, 0.8**2])))
    self.assertTrue(np.array_equal(self.cpm.cumulative_rewards, np.array([0, 1+1*0.5, 1+1*0.8])))

    cp2_rew, cp2_tau = self.cpm.reset_choicepoint("cp2")
    self.assertTrue(cp2_tau==2)
    self.assertTrue(cp2_rew==1+1*0.5)
    self.assertTrue(np.array_equal(self.cpm.discounts, np.array([1.0, 1.0, 0.8**2])))
    self.assertTrue(np.array_equal(self.cpm.cumulative_rewards, np.array([0, 0, 1+1*0.8])))

    self.cpm.distribute_reward(1)
    self.assertTrue(np.array_equal(self.cpm.discounts, np.array([1.0, 0.5, 0.8**3])))
    cp3_rew, cp3_tau = self.cpm.reset_choicepoint("cp3")
    self.assertTrue(cp3_tau==3)
    self.assertTrue(cp3_rew==1+1*0.8+1*0.8**2)
    self.assertTrue(np.array_equal(self.cpm.discounts, np.array([1.0, 0.5, 1.0])))
    self.assertTrue(np.array_equal(self.cpm.cumulative_rewards, np.array([1, 1, 0])))

  def test_eval(self):
    # use manual init cpm
    cpm = ChoicepointsManager(eval=True)
    for cp in self.cp_list:
      cpm.add_choicepoint(cp)
    self.assertTrue(np.array_equal(cpm.init_discounts, np.ones(len(self.cp_list))))
    cpm.reset()
    cpm.distribute_reward(1)
    cpm.distribute_reward(1)
    self.assertTrue(np.array_equal(cpm.discounts, np.ones(len(self.cp_list))))
    self.assertTrue(cpm.reset_choicepoint("cp1")[0]==2)
    self.assertTrue(cpm.reset_choicepoint("cp2")[0]==2)
    self.assertTrue(cpm.reset_choicepoint("cp3")[0]==2)

  def test_switch_eval(self):
    self.cpm.reset()
    self.cpm.distribute_reward(10)
    self.cpm.distribute_reward(10)
    self.cpm.distribute_reward(10)
    self.cpm.set_eval(True)
    self.cpm.reset()
    self.cpm.distribute_reward(1)
    self.cpm.distribute_reward(1)
    self.assertTrue(np.array_equal(self.cpm.discounts, np.ones(len(self.cp_list))))
    self.assertTrue(self.cpm.reset_choicepoint("cp1")[0]==2)
    self.assertTrue(self.cpm.reset_choicepoint("cp2")[0]==2)
    self.assertTrue(self.cpm.reset_choicepoint("cp3")[0]==2)

