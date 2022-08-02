import unittest
import pytest
import numpy as np
import copy
from numpy.random import default_rng

import gym
from gym import spaces

from pyham.ham import HAM, WrappedEnv, create_concat_joint_state_wrapped_env
from pyham.ham.utils import JointState

class TestBoxEnvFunctionality(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    cartpole_env = gym.make("CartPole-v1", new_step_api=False)
    discount = 0.99
    myham = HAM(discount)

    num_machines = 2
    reprs = np.eye(num_machines) # one-hot representation

    @myham.machine_with_repr(reprs[0])
    def top_loop(ham):
      while ham.is_alive:
        ham.CALL(double_action_machine)
        
    @myham.machine_with_repr(reprs[1])
    def double_action_machine(ham):
      binary_choice = int(ham.CALL_choice("binary"))
      ham.CALL_action(binary_choice)
      ham.CALL_action(binary_choice)
      return 0

    choice_space = spaces.Discrete(cartpole_env.action_space.n)
    machine_stack_cap = 3
    self.wrapped_env = create_concat_joint_state_wrapped_env(myham, 
                              cartpole_env, 
                              choice_space, 
                              initial_machine=top_loop,
                              np_pad_config = {"constant_values": 0},
                              machine_stack_cap=machine_stack_cap,
                              will_render=True)
    self.wrapped_env_no_render = create_concat_joint_state_wrapped_env((myham), 
                              cartpole_env, 
                              choice_space, 
                              initial_machine=top_loop,
                              np_pad_config = {"constant_values": 0},
                              machine_stack_cap=machine_stack_cap,
                              will_render=False)
    return super().setUp()

  @pytest.mark.timeout(3)
  def test_manual_init(self):
    cartpole_env = gym.make("CartPole-v1", new_step_api=False)
    discount = 0.99
    myham = HAM(discount)

    num_machines = 2
    repr_machine_cap = 3
    machine_stack_repr_shape = (repr_machine_cap*num_machines,)
    reprs = np.eye(num_machines) # one-hot representation
    machine_stack_high = np.ones(machine_stack_repr_shape[0])
    machine_stack_low = np.zeros(machine_stack_repr_shape[0])
    og_obsv_high = cartpole_env.observation_space.high
    og_obsv_low = cartpole_env.observation_space.low
    choice_space = spaces.Discrete(cartpole_env.action_space.n)
    js_space = spaces.Box(np.hstack((og_obsv_low, machine_stack_low)),
                                  np.hstack((og_obsv_high, machine_stack_high)),
                                  dtype = np.float32)

    @myham.machine_with_repr(reprs[0])
    def top_loop(ham):
      while ham.is_alive:
        ham.CALL(double_action_machine)
        
    @myham.machine_with_repr(reprs[1])
    def double_action_machine(ham):
      binary_choice = int(ham.CALL_choice("binary"))
      ham.CALL_action(binary_choice)
      ham.CALL_action(binary_choice)
      return 0

    def js2obsv(js: JointState):
      machine_stack_repr = np.hstack(js.m[-repr_machine_cap:])
      padding = max(0, machine_stack_repr_shape[0]-len(machine_stack_repr))
      machine_stack_repr = np.pad(machine_stack_repr, (padding,0), 'constant', constant_values=(0,0))
      js_repr = np.float32(np.hstack((js.s,machine_stack_repr)))
      return js_repr

    wrapped_env = WrappedEnv(myham, 
                              cartpole_env, 
                              choice_space, 
                              js_space, 
                              js2obsv, 
                              initial_machine=top_loop,
                              will_render=True)
    self.test_rendering_and_running(wrapped_env)
    self.test_rendering_and_running(wrapped_env)

  @pytest.mark.timeout(3)
  def test_running(self, env=None):
    if env is None:
      env = self.wrapped_env_no_render
    obsv = env.reset(seed=0)
    self.assertTrue(len(obsv)==10)
    self.assertTrue(np.array_equal(np.array([0,0,1,0,0,1]), obsv[4:]))
    obsv, reward, done, info = env.step(0)
    self.assertTrue(np.array_equal(np.array([0,0,1,0,0,1]), obsv[4:]))
    self.assertTrue(reward == 1+0.99)
    self.assertFalse(done)
    self.assertTrue(info['next_choice_point'] == "binary")
    while not done:
      obsv, reward, done, info = env.step(0)
    self.assertTrue(done)

  @pytest.mark.timeout(3)
  def test_rendering_and_running(self, env=None):
    if env is None:
      env = self.wrapped_env
    obsv = env.reset(seed=0)
    frames = env.render()
    self.assertTrue(len(frames)==1)
    self.assertEqual(frames[0].shape, (400, 600, 3))
    
    obsv, reward, done, info = env.step(0)
    frames = env.render()
    self.assertEqual(len(frames), 2)
    for frame in frames:
      self.assertEqual(frame.shape, (400, 600, 3))
    
    # also test normal running
    self.test_running(self.wrapped_env)
    
  @pytest.mark.timeout(3)
  def test_turning_render_mode_on_off(self):
    self.wrapped_env.set_render_mode(True)
    self.test_rendering_and_running(self.wrapped_env)
    self.wrapped_env.set_render_mode(False)
    self.test_running(self.wrapped_env)
    

class TestMultiDiscreteEnvFunctionality(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    taxi_env = DecodedMultiDiscreteWrapper(gym.make("Taxi-v3", new_step_api=False), [5,5,5,4])
    discount = 0.99
    myham = HAM(discount)

    num_machines = 2
    reprs = np.eye(num_machines) # one-hot representation

    @myham.machine_with_repr(reprs[0])
    def top_loop(ham):
      while ham.is_alive:
        ham.CALL(nav_machine)
        
    @myham.machine_with_repr(reprs[1])
    def nav_machine(ham):
      choice = int(ham.CALL_choice("nav"))
      ham.CALL_action(choice)
      return 0

    choice_space = spaces.Discrete(taxi_env.action_space.n)
    machine_stack_cap = 3
    self.wrapped_env = create_concat_joint_state_wrapped_env(myham, 
                              taxi_env, 
                              choice_space, 
                              initial_machine=top_loop,
                              np_pad_config = {"constant_values": 0},
                              machine_stack_cap=machine_stack_cap,
                              will_render=False)
    return super().setUp()

  @pytest.mark.timeout(3)
  def test_running(self, env=None):
    if env is None:
      env = self.wrapped_env
    obsv = env.reset(seed=0)
    self.assertTrue(np.array_equal(env.observation_space.nvec, np.array([5,5,5,4,2,2,2,2,2,2])))
    self.assertTrue(env.observation_space.contains(obsv))
    self.assertTrue(len(obsv)==10)
    self.assertTrue(np.array_equal(np.array([0,0,1,0,0,1]), obsv[4:]))
    obsv, reward, done, info = env.step(0)
    self.assertTrue(np.array_equal(np.array([0,0,1,0,0,1]), obsv[4:]))
    self.assertTrue(info['next_choice_point'] == "nav")

class TestVariousHAMs(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    self.cartpole_env = gym.make("CartPole-v1", new_step_api=False)
    self.discount = 0.99
    return super().setUp()

  def test_triple_choice(self):
    rng = default_rng(42)
    myham = HAM(self.discount)

    num_machines = 2
    reprs = np.eye(num_machines) # one-hot representation

    @myham.machine_with_repr(reprs[0])
    def top_loop(ham):
      while ham.is_alive:
        ham.CALL(double_action_machine)
        
    @myham.machine_with_repr(reprs[1])
    def double_action_machine(ham):
      choice = int(ham.CALL_choice("012"))
      if choice<2:
        ham.CALL_action(choice)
        ham.CALL_action(choice)
      return 0

    choice_space = spaces.Discrete(3)
    machine_stack_cap = 3
    wrapped_env = create_concat_joint_state_wrapped_env(myham, 
                              self.cartpole_env, 
                              choice_space, 
                              initial_machine=top_loop,
                              np_pad_config = {"constant_values": 0},
                              machine_stack_cap=machine_stack_cap,
                              will_render=False)
    self.assertEqual(wrapped_env.action_space.n, 3)
    wrapped_env.reset(seed=0)
    done = False
    while not done:
      obsv, reward, done, info = wrapped_env.step(rng.integers(3))
    self.assertTrue(done)

## AUXILIARY

from typing import Union, List

class DecodedMultiDiscreteWrapper(gym.ObservationWrapper):
  def __init__(self, env: gym.Env, nvec: Union[List[int], np.ndarray]):
    super().__init__(env, new_step_api=True)
    self.obsv_decoder = lambda x: list(env.decode(x))
    self.observation_space = spaces.MultiDiscrete(nvec)

  def observation(self, observation):
    return self.obsv_decoder(observation)