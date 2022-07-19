from typing import Optional, Union, Callable, Any
import copy
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