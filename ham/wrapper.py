from typing import Optional, Union, Callable
import gym
from gym import spaces

from ham import HAM
from ham.utils import JointState


class WrappedEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ["rgb_array"]}

  def __init__(self, 
    ham:HAM, 
    env:gym.Env, 
    choice_space: spaces.Space, 
    joint_state_space: spaces.Space,
    initial_machine: Union[Callable, str]
  ):
    super(WrappedEnv, self).__init__()
    self.ham = ham
    self.env = env
    self.action_space = choice_space
    self.observation_space = joint_state_space
    self.initial_machine = initial_machine


  def joint_state_to_observation(self, joint_state: JointState):
    observation = None # TODO fill this

    assert self.observation_space.contains(observation), "Invalid observation"
    return observation

  def step(self, choice):
    assert self.ham.is_alive, "HAMs is not started or has stopped. Try reset env."
    assert self.action_space.contains(choice), "Invalid choice"

    joint_state, reward, done, info = self.ham.step(choice)
    observation = self.joint_state_to_observation(joint_state)
    return observation, reward, done, info
    

  def reset(self, seed:Optional[int]=None):
    cur_obsv = self.env.reset(seed=seed)
    self.ham.episodic_reset(cur_obsv)
    joint_state, reward, done, info = self.ham.start(self.initial_machine)
    return self.joint_state_to_observation(joint_state)


  def render(self, mode="rgb_array"):
    # TODO
    pass

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