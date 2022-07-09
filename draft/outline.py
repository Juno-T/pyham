from __future__ import annotations # for forward reference of HAM and type hint
import copy
from typing import Tuple, NamedTuple, Type, Any, Callable, Union

# TODO Test transition/choice point handler/ reward system
# TODO*** Machine's customizable parameters. Urgent, for repetative actions.
# TODO refactor
# TODO HAMQ-INT internal transition skip?

class JointState(NamedTuple):
  s: Any # Environment pure state
  m: Any # HAM's machine stack

class Transition(NamedTuple):
  s_tm1: JointState
  a_tm1: Any # choice
  r_t: float # cumulative reward between choice point
  s_t: JointState
  done: int # 1 if done, else 0


class HAM:
  def __init__(
    self, 
    action_executor: Callable[[Any], Tuple], 
    transition_handler: Callable, 
    reward_discount: float = 0.9,
    verbose: bool = False
  ):
    """
      Parameters:
        action_executor: A function that handle action execution. Must return (obsv, reward, done, info). Can be as simple as gym's env.step.
        transition_handler: A function that handle transition upon new transition is created (when there is a choice point).
    """
    self.action_executor = action_executor
    self.transition_handler = transition_handler
    self.reward_discount = reward_discount
    self.machines = {}
    self.machine_count=0
    self.verbose =verbose

  def set_observation(self, obsv):
    self._current_observation=obsv

  def episodic_reset(self, current_observation):
    """
      Must be called before each episode
    """
    self._current_observation = current_observation
    self._machine_stack = []
    self._cumulative_reward = 0.
    self._cumulative_discount = 1.
    self._previous_choice_point = None
    self._tmp_return = None
    

  def choice_point_handler(self, choice, done=False):
    joint_state = JointState(
      s=self._current_observation,
      m=copy.deepcopy(self._machine_stack)
    )
    transition = None
    if self._previous_choice_point is not None:
      prev_joint_state, prev_choice = self._previous_choice_point
      transition = Transition(
        s_tm1 = prev_joint_state,
        a_tm1 = prev_choice,
        r_t = self._cumulative_reward,
        s_t = joint_state,
        done = 1 if done else 0
      )
    self._previous_choice_point = (joint_state, choice)
    if transition is not None:
      self.transition_handler(transition)
    self._cumulative_reward=0.
    self._cumulative_discount = 1.
    

  def _add_machine(self, name, machine, representation=None):
    """
      Registering machine created by self._create_*_machine
    """
    if self.verbose:
      print(name+": \r")
    next(machine)
    if representation is None:
      representation = self.machine_count
    self.machine_count+=1
    self.machines[name]={
      "machine": machine,
      "representation": representation,
    }

  def functional_machine(self, func: Callable[[Type[HAM], Any], Any], representation=None):
    """
      A convinent decorator for creating and registering functional machine
    """
    machine_name = func.__name__
    machine = self._create_functional_machine(func)
    self._add_machine(machine_name, machine, representation)
    return func

  def action_machine(self, action_selector: Callable[[Type[HAM], Any], Any], representation=None):
    """
      A convinent decorator for creating and registering action machine
    """
    machine_name = action_selector.__name__
    machine = self._create_action_machine(action_selector)
    self._add_machine(machine_name, machine, representation)
    return action_selector

  def learnable_choice_machine(self, selector: Callable[[Type[HAM], Any], Any], representation=None):
    """
      A convinent decorator for creating and registering learnable choice machine
    """
    machine_name = selector.__name__
    machine = self._create_learnable_choice_machine(selector)
    self._add_machine(machine_name, machine, representation)
    return selector

  def _create_functional_machine(self, func: Callable[[Type[HAM], Any], Any]):
    """
      Parameters:
        func: A pure python function
    """
    if self.verbose:
      print("Functional machine initiated")
    while True:
      smth = yield # smth should be unified
      ret = func(self, smth)
      self._tmp_return = ret

  def _create_action_machine(self, action_selector: Callable[[Type[HAM], Any], Any]):
    """
      Parameters:
        action_selector: A pure function that return an action given the input
    """
    if self.verbose:
      print("Action machine initiated")
    while True:
      smth = yield # smth should be unified
      action = action_selector(self,smth)
      obsv, reward, done, info = self.action_executor(action)
      self.set_observation(obsv)
      self._cumulative_reward+=reward*self._cumulative_discount
      self._cumulative_discount*=self.reward_discount
      if done:
        self.choice_point_handler(None, done = True)
      self._tmp_return = obsv

  def _create_learnable_choice_machine(self, selector: Callable[[Type[HAM], Any], Any]):
    """
      Parameters:
        selector: A pure function that return an option given the input
    """
    if self.verbose:
      print("Learnable Choice machine initiated")
    while True:
      smth = yield
      choice = selector(self, smth)
      self.choice_point_handler(choice)
      self._tmp_return = choice

  def CALL(self, machine: Union[str, Callable]):
    """
      A method to CALL other registered machine.
      Parameters:
        machine: Machine's original function or it's name.
    """
    if isinstance(machine, str):
      machine_name = machine
    else:
      machine_name = machine.__name__
    assert(machine_name in self.machines)
    try:
      self._machine_stack.append(self.machines[machine_name]["representation"])
      self.machines[machine_name]["machine"].send(self._current_observation) # TODO this will define all the smth
      self._machine_stack.pop()
    except StopIteration:
      print("STOP ITERATION??")
      self._tmp_return = None
    return self._tmp_return


  
