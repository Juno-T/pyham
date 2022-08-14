try:
  from ray.rllib.env.multi_agent_env import MultiAgentEnv
except:
  raise("Multi-choice environment requires `ray[rllib]`.")

from typing import Optional, Union, Callable, Any, Dict
import copy
import gym
from gym import spaces

from ..ham import HAM
from ..utils import JointState


class MultiChoiceTypeEnv(MultiAgentEnv):
  """
    Custom Wrapped Environment which converts HAM into rllib's MultiAgentEnv environment.
  """
  metadata = {'render_modes': ["rgb_array"]}

  def __init__(self, 
    ham:HAM, 
    env:gym.Env,
    joint_state_space: spaces.Space,
    joint_state_to_representation: Callable[[JointState], Any],
    initial_machine: Union[Callable, str],
    initial_args: Union[list, tuple]=[],
    eval: bool = False,
    will_render: bool = False,
  ):
    """
      Parameters:
        ham: Instantiated ham with machines registered
        env: gym environment to use
        joint_state_space: Joint state representation space
        joint_state_to_representation: A function that convert `JointState` to joint state representation accourding to `joint_state_space`
        initial_machine: The top level machine to start the HAM with.
        eval: whether to instantiate env for evaluation or not.
        will_render: If true, pre-render every frames even if `render()` is not being called. Must be set to true if `render()` method is expected to be called.
    """
    super().__init__()
    self.ham = copy.deepcopy(ham)
    self.ham.set_eval(eval)
    self.env = env
    self.observation_space = joint_state_space
    self.joint_state_to_representation = joint_state_to_representation
    self.initial_machine = initial_machine
    self.initial_args = initial_args
    self.eval = eval # TODO
    self.will_render = will_render

    self._all_not_done = {
      cp.name: False
      for cp in self.ham.cpm
    }
    self._all_not_done["__all__"] = False
    self._all_done = {
      cp.name: True
      for cp in self.ham.cpm
    }
    self._all_done["__all__"] = True
    
    
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
      self.ham.terminate()
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

    self.render_stack=[]
    joint_state, reward, done, info = self.ham.step(choice)
    self.actual_ep_len += self.ham._tau
    next_choicepoint_name = info["next_choicepoint_name"]
    if done:
      obsv, reward = self.create_last_step(next_choicepoint_name, joint_state, reward)
    else:
      js_repr = self.joint_state_to_representation(joint_state)
      obsv = {next_choicepoint_name: js_repr}
      reward = {next_choicepoint_name: reward}
    done = self._all_done if done else self._all_not_done
    return obsv, reward, done, info
    
  def reset(self, seed:Optional[int]=None):
    """
      Reset api. Must be called before each episode.
      Parameters:
        seed: seed value
      Return:
        Initial joint state representation
    """
    try:
      cur_obsv = self.env.reset(seed=seed)
    except:
      self.env.seed(seed)
      cur_obsv = self.env.reset()
    self.render_stack = []
    if self.will_render:
      rendered_frame = self.env.render(mode="rgb_array")
      self.render_stack.append(rendered_frame)
    self.ham.episodic_reset(cur_obsv)
    joint_state, reward, done, info = self.ham.start(self.initial_machine, args=self.initial_args)
    self.actual_ep_len = joint_state.tau
    js_repr = self.joint_state_to_representation(joint_state)
    assert self.observation_space.contains(js_repr), f"Invalid `JointState` to observation conversion."
    return {
      info["next_choicepoint_name"]: js_repr
    }

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

  def seed(self, seed: Optional[int] = None):
    if seed is None:
      return
    self.env.seed(seed)

  # DUCKTAPE # TODO move this to HAM
  def create_last_step(self, next_choicepoint_name, joint_state, reward):
    joint_states = {
      cp.name: self.joint_state_to_representation(
        joint_state._replace(tau=self.ham.cpm.tau[i])
      )
      for i, cp in enumerate(self.ham.cpm)
    }
    joint_states[next_choicepoint_name]=self.joint_state_to_representation(joint_state)

    rewards = {
      cp.name: self.ham.cpm.cumulative_rewards[i]
      for i, cp in enumerate(self.ham.cpm)
    }
    rewards[next_choicepoint_name]=reward
    return joint_states, rewards

  def rllib_policies_config(self) -> Dict[str, Dict[str, Any]]:
    """
      Returns rllib's multi agent `policies config` for each choicepoint in the form of PolicySpec arguments:
      {
        "choicepoint1's_name": {
          "policy_class": None,
          "observation_space": ham's joint_state_space,
          "action_space": choicepoint1's choice_space, 
        },
        ...
      }
    """
    policies = {
      cp.name:{
        "policy_class":None,  # infer automatically from Trainer
        "observation_space": self.observation_space,  # ham's joint_state_space
        "action_space": cp.choice_space,  # choicepoint1's choice_space
      }
      for cp in self.ham.cpm
    }
    return policies