import unittest
import pytest
import numpy as np
import copy
from numpy.random import default_rng

import gym
from gym import spaces

from pyham import HAM
from pyham.wrappers.multi_choice import MultiChoiceTypeEnv
from pyham.wrappers.helpers import create_concat_joint_state_env
from pyham.utils import JointState
from pyham.integration.gym_wrappers import DecodedMultiDiscreteWrapper

class TestBoxEnvFunctionality(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    cartpole_env = gym.make("CartPole-v1")
    myham = HAM()

    num_machines = 2

    cp1 = myham.choicepoint("cp1", spaces.Discrete(3), discount=0.1)
    cp2 = myham.choicepoint("cp2", spaces.Discrete(2), discount=0.2)
    
    @myham.machine
    def top_loop(ham: HAM):
      while ham.is_alive:
        rep = ham.CALL_choice(cp1)
        ham.CALL(rep_action_machine, (rep,))
        
    @myham.machine
    def rep_action_machine(ham: HAM, rep):
      action = ham.CALL_choice(cp2)
      for _ in range(rep):
        ham.CALL_action(action)

    machine_stack_cap = 2
    self.wrapped_env = create_concat_joint_state_env(myham, 
                              cartpole_env,
                              initial_machine=top_loop,
                              np_pad_config = {"constant_values": 0},
                              machine_stack_cap=machine_stack_cap,
                              will_render=True)
    self.wrapped_env_no_render = create_concat_joint_state_env((myham), 
                              cartpole_env, 
                              initial_machine=top_loop,
                              np_pad_config = {"constant_values": 0},
                              machine_stack_cap=machine_stack_cap,
                              will_render=False)
    return super().setUp()

  # @pytest.mark.timeout(3)
  # def test_manual_init(self):
  #   pass
  def test_verify_policies_config(self):
    config = self.wrapped_env_no_render.rllib_policies_config()
    self.assertTrue("cp1" in config)
    self.assertTrue("cp2" in config)
    self.assertTrue(config["cp1"]["observation_space"].shape==(8,))
    self.assertTrue(config["cp2"]["observation_space"].shape==(8,))
    self.assertTrue(config["cp1"]["action_space"].n==3)
    self.assertTrue(config["cp2"]["action_space"].n==2)

  @pytest.mark.timeout(3)
  def test_running(self, env=None):
    if env is None:
      env = self.wrapped_env_no_render
    obsv = env.reset(seed=0)
    self.assertTrue(isinstance(obsv, dict))
    self.assertTrue("cp1" in obsv)
    self.assertTrue(len(obsv["cp1"])==8)
    self.assertTrue(np.array_equal(np.array([0,0,1,0]), obsv["cp1"][4:]))
    obsv, reward, done, info = env.step(2) # rep
    self.assertTrue("cp2" in obsv)
    self.assertTrue("cp2" in reward)
    self.assertTrue(len(obsv["cp2"])==8)
    self.assertTrue(np.array_equal(np.array([1,0,0,1]), obsv["cp2"][4:]))
    self.assertTrue(reward["cp2"] == 0)
    self.assertTrue(env.actual_ep_len==0)
    self.assertTrue((not done["cp1"]) and (not done["cp2"]) and (not done["__all__"]))
    obsv, reward, done, info = env.step(0) # direction
    self.assertTrue("cp1" in obsv)
    self.assertTrue("cp1" in reward)
    self.assertTrue(len(obsv["cp1"])==8)
    self.assertTrue(np.array_equal(np.array([0,0,1,0]), obsv["cp1"][4:]))
    self.assertTrue(reward["cp1"] == 1+1*0.1)
    self.assertTrue((not done["cp1"]) and (not done["cp2"]) and (not done["__all__"]))
    while not done["__all__"]:
      obsv, reward, done, info = env.step(1)
    self.assertTrue("cp1" in obsv)
    self.assertTrue("cp1" in reward)
    self.assertTrue("cp2" in obsv)
    self.assertTrue("cp2" in reward)
    self.assertTrue(done["__all__"])

#   @pytest.mark.timeout(3)
#   def test_rendering_and_running(self, env=None):
#     if env is None:
#       env = self.wrapped_env
#     obsv = env.reset(seed=0)
#     frames = env.render()
#     self.assertTrue(len(frames)==1)
#     self.assertEqual(frames[0].shape, (400, 600, 3))
    
#     obsv, reward, done, info = env.step(0)
#     frames = env.render()
#     self.assertEqual(len(frames), 2)
#     for frame in frames:
#       self.assertEqual(frame.shape, (400, 600, 3))
#     env.close()
#     # also test normal running
#     self.test_running(env)
#     env.close()
    
#   @pytest.mark.timeout(3)
#   def test_turning_render_mode_on_off(self):
#     self.wrapped_env.set_render_mode(True)
#     self.test_rendering_and_running(self.wrapped_env)
#     self.wrapped_env.set_render_mode(False)
#     self.test_running(self.wrapped_env)
#     self.wrapped_env.close()
    

# class TestMultiDiscreteEnvFunctionality(unittest.TestCase):
#   @classmethod
#   def setUpClass(cls) -> None:
#     return super().setUpClass()

#   def setUp(self) -> None:
#     taxi_env = DecodedMultiDiscreteWrapper(gym.make("Taxi-v3"), [5,5,5,4])
    
#     myham = HAM()

#     num_machines = 2
#     reprs = np.eye(num_machines) # one-hot representation

#     nav = myham.choicepoint("nav", spaces.Discrete(4), discount = 0.99)

#     @myham.machine_with_repr(reprs[0])
#     def top_loop(ham):
#       while ham.is_alive:
#         ham.CALL(nav_machine)
        
#     @myham.machine_with_repr(reprs[1])
#     def nav_machine(ham):
#       choice = int(ham.CALL_choice(nav))
#       ham.CALL_action(choice)
#       return 0

#     machine_stack_cap = 3
#     self.wrapped_env = create_concat_joint_state_SingleChoiceTypeEnv(myham, 
#                               taxi_env, 
#                               initial_machine=top_loop,
#                               np_pad_config = {"constant_values": 0},
#                               machine_stack_cap=machine_stack_cap,
#                               will_render=False)
#     return super().setUp()

#   @pytest.mark.timeout(3)
#   def test_running(self, env=None):
#     if env is None:
#       env = self.wrapped_env
#     obsv = env.reset(seed=0)
#     self.assertTrue(np.array_equal(env.observation_space.nvec, np.array([5,5,5,4,2,2,2,2,2,2])))
#     self.assertTrue(env.observation_space.contains(obsv))
#     self.assertTrue(len(obsv)==10)
#     self.assertTrue(np.array_equal(np.array([0,0,1,0,0,1]), obsv[4:]))
#     obsv, reward, done, info = env.step(0)
#     self.assertTrue(np.array_equal(np.array([0,0,1,0,0,1]), obsv[4:]))
#     self.assertTrue(info['next_choicepoint_name'] == "nav")
#     env.close()

# class TestVariousHAMs(unittest.TestCase):
#   @classmethod
#   def setUpClass(cls) -> None:
#     return super().setUpClass()

#   def setUp(self) -> None:
#     self.cartpole_env = gym.make("CartPole-v1")
#     self.discount = 0.99
#     return super().setUp()

#   def test_triple_choice(self):
#     rng = default_rng(42)
#     myham = HAM()

#     num_machines = 2
#     reprs = np.eye(num_machines) # one-hot representation

#     triplechoice = myham.choicepoint("triplechoice", spaces.Discrete(3), discount=self.discount)
#     @myham.machine_with_repr(reprs[0])
#     def top_loop(ham):
#       while ham.is_alive:
#         ham.CALL(double_action_machine)
        
#     @myham.machine_with_repr(reprs[1])
#     def double_action_machine(ham):
#       choice = int(ham.CALL_choice(triplechoice))
#       if choice<2:
#         ham.CALL_action(choice)
#         ham.CALL_action(choice)
#       return 0

#     machine_stack_cap = 3
#     wrapped_env = create_concat_joint_state_SingleChoiceTypeEnv(myham, 
#                               self.cartpole_env, 
#                               initial_machine=top_loop,
#                               np_pad_config = {"constant_values": 0},
#                               machine_stack_cap=machine_stack_cap,
#                               will_render=False)
#     self.assertEqual(wrapped_env.action_space.n, 3)
#     wrapped_env.reset(seed=0)
#     done = False
#     while not done:
#       obsv, reward, done, info = wrapped_env.step(rng.integers(3))
#     self.assertTrue(done)

#   def test_default_repr_ham(self):
#     rng = default_rng(42)
#     myham = HAM() # default representation=onehot


#     triplechoice = myham.choicepoint("triplechoice", spaces.Discrete(3), discount=self.discount)
#     @myham.machine
#     def top_loop(ham):
#       while ham.is_alive:
#         ham.CALL(double_action_machine)
        
#     @myham.machine
#     def double_action_machine(ham):
#       choice = int(ham.CALL_choice("triplechoice"))
#       if choice<2:
#         ham.CALL_action(choice)
#         ham.CALL_action(choice)
#       return 0

#     machine_stack_cap = 3
#     wrapped_env = create_concat_joint_state_SingleChoiceTypeEnv(myham, 
#                               self.cartpole_env, 
#                               initial_machine=top_loop,
#                               np_pad_config = {"constant_values": 0},
#                               machine_stack_cap=machine_stack_cap,
#                               will_render=False)
#     self.assertEqual(wrapped_env.action_space.n, 3)
#     wrapped_env.reset(seed=0)
#     done = False
#     while not done:
#       obsv, reward, done, info = wrapped_env.step(rng.integers(3))
#     self.assertTrue(done)