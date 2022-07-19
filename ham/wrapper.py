from typing import Optional, Union, Callable, Any
import copy
import numpy as np
import gym
from gym import spaces

from ham import HAM
from ham.utils import JointState


class WrappedEnv(gym.Env):
  """
    Custom Wrapped Environment which converts HAM into gym-like environment.
  """
  metadata = {'render_modes': ["rgb_array"]}

  def __init__(self, 
    ham:HAM, 
    env:gym.Env, 
    choice_space: spaces.Space, 
    joint_state_space: spaces.Space,
    joint_state_to_representation: Callable[[JointState], Any],
    initial_machine: Union[Callable, str],
    initial_args = [],
    will_render: bool = False,
  ):
    """
      Parameters:
        ham: Instantiated ham with machines registered
        env: gym environment to use
        choice_space: Domain of choices in the registered machines.
        joint_state_space: Joint state representation space
        joint_state_to_representation: A function that convert `JointState` to joint state representation accourding to `joint_state_space`
        initial_machine: The top level machine to start the HAM with.
        will_render: If true, pre-render every frames even if `render()` is not being called. Must be set to true if `render()` method is expected to be called.
    """
    super(WrappedEnv, self).__init__()
    self.ham = copy.deepcopy(ham)
    self.env = env
    self.action_space = choice_space
    self.observation_space = joint_state_space
    self.joint_state_to_representation = joint_state_to_representation
    self.initial_machine = initial_machine
    self.initial_args = initial_args
    self.will_render = will_render
    
    if self.will_render:
      self.ham.action_executor = self._create_wrapped_action_executor()
    else:
      self.ham.action_executor = self.env.step
    self.render_stack = []

  def set_render_mode(self, mode: bool):
    """
      Turn render mode on or off after instantiation.
      Parameters:
        mode: `bool` True means on, False means off.
    """
    if self.ham.is_alive:
      raise("Cannot set render mode while HAM is running.")
      return None
    self.render_stack=[]
    if mode==True:
      self.will_render=True
      self.ham.action_executor = self._create_wrapped_action_executor()
    else:
      self.will_render = False
      self.ham.action_executor=self.env.step

  def _create_wrapped_action_executor(self):
    def _action_executor(*args, **kwargs):
      ret = self.env.step(*args, **kwargs)
      rendered_frame = self.env.render(mode="rgb_array")
      self.render_stack.append(rendered_frame)
      return ret
    return _action_executor

  def step(self, choice):
    """
      gym-like step api
      Parameters:
        choice: choice to be executed at choice point
      Return:
        A 4 items tuple:
          joint state representation: A representation of joint state processed with `joint_state_to_representation`.
          cumulative reward: Cumulative reward
          done: Environment done or ham done.
          info: dictionary with extra info, e.g. info['next_choice_point']
    """
    assert self.ham.is_alive, "HAMs is not started or has stopped. Try reset env."
    assert self.action_space.contains(choice), "Invalid choice"

    self.render_stack=[]
    joint_state, reward, done, info = self.ham.step(choice)
    js_repr = self.joint_state_to_representation(joint_state)
    assert self.observation_space.contains(js_repr), "Invalid `JointState` to observation conversion."
    return js_repr, reward, done, info
    
  def reset(self, seed:Optional[int]=None):
    """
      Reset api. Must be called before each episode.
      Parameters:
        seed: seed value
      Return:
        Initial joint state representation
    """
    cur_obsv = self.env.reset(seed=seed)
    self.render_stack = []
    if self.will_render:
      rendered_frame = self.env.render(mode="rgb_array")
      self.render_stack.append(rendered_frame)
    self.ham.episodic_reset(cur_obsv)
    joint_state, reward, done, info = self.ham.start(self.initial_machine, args=self.initial_args)
    js_repr = self.joint_state_to_representation(joint_state)
    assert self.observation_space.contains(js_repr), f"Invalid `JointState` to observation conversion."
    return js_repr

  def render(self, mode="rgb_array"):
    """
      Render frames
      Parameter:
        mode: only support "rgb_array" for now.
      Return:
        A list of rendered frames in numpy array since previous choice point.
    """
    assert mode in self.metadata["render_modes"], "Invalid rendering mode"
    return self.render_stack

  def close(self):
    """
      Close environment & terminate ham.
    """
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
                                    initial_args = [],
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
  js_space = spaces.Box(np.hstack((og_obsv_low, machine_stack_low)),
                                np.hstack((og_obsv_high, machine_stack_high)),
                                dtype = dtype)
  def js2repr(js: JointState):
    machine_stack_repr = np.hstack(js.m[-machine_stack_cap:])
    padding = max(0, machine_stack_repr_shape[0]-len(machine_stack_repr))
    machine_stack_repr = np.pad(machine_stack_repr, (padding,0), **np_pad_config)
    js_repr = np.float32(np.hstack((js.s,machine_stack_repr)))
    return js_repr
  return WrappedEnv(ham, 
                    env, 
                    choice_space, 
                    js_space, 
                    js2repr, 
                    initial_machine=initial_machine,
                    initial_args=initial_args,
                    will_render=will_render)
  