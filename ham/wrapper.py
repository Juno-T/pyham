from typing import Optional, Union, Callable, Any
import copy
import numpy as np
import gym
from gym import spaces

from ham import HAM
from ham.utils import JointState


class WrappedEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render_modes': ["rgb_array"]}

  def __init__(self, 
    ham:HAM, 
    env:gym.Env, 
    choice_space: spaces.Space, 
    joint_state_space: spaces.Space,
    joint_state_to_observation: Callable[[JointState], Any],
    initial_machine: Union[Callable, str],
    will_render: bool = False, # If true, pre-render every frames even if `render()` is not called.
  ):
    super(WrappedEnv, self).__init__()
    self.ham = copy.deepcopy(ham)
    self.env = env
    self.action_space = choice_space
    self.observation_space = joint_state_space
    self.joint_state_to_observation = joint_state_to_observation
    self.initial_machine = initial_machine
    self.will_render = will_render
    
    if self.will_render:
      self.ham.action_executor = self.create_wrapped_action_executor()
    else:
      self.ham.action_executor = self.env.step
    self.render_stack = []

  def create_wrapped_action_executor(self):
    def action_executor(*args, **kwargs):
      ret = self.env.step(*args, **kwargs)
      rendered_frame = self.env.render(mode="rgb_array")
      self.render_stack.append(rendered_frame)
      return ret
    return action_executor

  def step(self, choice):
    assert self.ham.is_alive, "HAMs is not started or has stopped. Try reset env."
    assert self.action_space.contains(choice), "Invalid choice"

    self.render_stack=[]
    joint_state, reward, done, info = self.ham.step(choice)
    observation = self.joint_state_to_observation(joint_state)
    assert self.observation_space.contains(observation), "Invalid `JointState` to observation conversion."
    return observation, reward, done, info
    

  def reset(self, seed:Optional[int]=None):
    cur_obsv = self.env.reset(seed=seed)
    self.render_stack = []
    if self.will_render:
      rendered_frame = self.env.render(mode="rgb_array")
      self.render_stack.append(rendered_frame)
    self.ham.episodic_reset(cur_obsv)
    joint_state, reward, done, info = self.ham.start(self.initial_machine)
    obsv = self.joint_state_to_observation(joint_state)
    assert self.observation_space.contains(obsv), f"Invalid `JointState` to observation conversion."
    return obsv

  def render(self, mode="rgb_array"):
    assert mode in self.metadata["render_modes"], "Invalid rendering mode"
    return self.render_stack

  def close(self):
    try:
      self.env.close()
    except:
      pass
    if self.ham.is_alive:
      try:
        self.ham.terminate()
      except:
        pass

def create_concat_joint_state_wrapped_env(ham: HAM, 
                                    env: gym.Env, 
                                    choice_space: spaces.Space,
                                    initial_machine: Union[Callable, str],
                                    np_pad_config: dict,
                                    machine_stack_cap: int = 1,
                                    dtype=np.float32,
                                    will_render=False
                                    ):
  """
    Currently support only env with continuous observation space.

    Parameters:
      ham: HAM
      env: gym.Env
      choice_space: choice space designed in ham
      np_pad_config: dictionary containing numpy.pad's mode and the reset of kwargs.
        e.g. 
        np_pad_config = {
          "mode": "constant", # (default)
          "constant_values": 0
        }
      machine_stack_cap: Number of top most machines to include in the joint state representation, pad if needed.
    Return:
      Wrapped env with joint state representation defined as concatenated numpy array between original env's observation space and fixed length of machine stack representations.
  """

  def _obj_len(obj):
    if hasattr(obj, '__len__'):
      return len(obj)
    else:
      return 1

  # num_machines = ham.machine_count
  repr_length = None
  for machine_name, machine in ham.machines.items():
    repr = machine['representation']
    if repr_length is None:
      repr_length = _obj_len(repr)
    elif repr_length != _obj_len(repr):
      raise("Unable to create joint state representation due to inconsistent representation length.")

  machine_stack_repr_shape = (machine_stack_cap*repr_length,)
  machine_stack_high = np.ones(machine_stack_repr_shape[0])
  machine_stack_low = np.zeros(machine_stack_repr_shape[0])
  og_obsv_high = env.observation_space.high
  og_obsv_low = env.observation_space.low
  choice_space = spaces.Discrete(env.action_space.n)
  js_space = spaces.Box(np.hstack((og_obsv_low, machine_stack_low)),
                                np.hstack((og_obsv_high, machine_stack_high)),
                                dtype = dtype)
  def js2obsv(js: JointState):
    machine_stack_repr = np.hstack(js.m[-machine_stack_cap:])
    padding = max(0, machine_stack_repr_shape[0]-len(machine_stack_repr))
    machine_stack_repr = np.pad(machine_stack_repr, (padding,0), **np_pad_config)
    js_repr = np.float32(np.hstack((js.s,machine_stack_repr)))
    return js_repr
  return WrappedEnv(ham, 
                    env, 
                    choice_space, 
                    js_space, 
                    js2obsv, 
                    initial_machine=initial_machine,
                    will_render=will_render)
  