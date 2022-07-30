import gym
from gym import spaces
import numpy as np
from typing import Union, List

class DecodedMultiDiscreteWrapper(gym.ObservationWrapper):
  def __init__(self, env: gym.Env, nvec: Union[List[int], np.ndarray]):
    super().__init__(env, new_step_api=True)
    self.obsv_decoder = lambda x: list(env.decode(x))
    self.observation_space = spaces.MultiDiscrete(nvec)

  def observation(self, observation):
    return self.obsv_decoder(observation)
  
class MultiDiscrete2NormBoxWrapper(gym.ObservationWrapper):
  def __init__(self, env: gym.Env, dtype = np.float64):
    super().__init__(env, new_step_api=True)
    self.sp_range = env.observation_space.nvec
    self.dtype = dtype
    self.mean = self.dtype(self.sp_range-1)/2.
    
    sp_mx = self.transform(self.sp_range-1.)
    self.observation_space = spaces.Box(-1, 1, shape=sp_mx.shape, dtype=self.dtype)

  def observation(self, observation):
    return self.transform(observation)

  def transform(self, obsv):
    obsv = self.dtype(obsv)-self.mean
    return np.clip(obsv/self.mean, -1., 1.)