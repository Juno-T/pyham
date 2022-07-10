from __future__ import annotations # for forward reference of HAM and type hint
import copy
from typing import Tuple, NamedTuple, Type, Any, Callable, Union

# TODO refactor
# TODO What if choice machines have different kind of choice returned?
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
    transition_handler: Callable[[Type[Transition],]], 
    reward_discount: float = 0.9,
    verbose: bool = False
  ):
    """
      A hierarchical of abstract machine (HAM) class.
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

  def set_observation(self, current_observation):
    """
      Set current observation.
      Parameters:
        current_observation: An observation to be set.
    """
    self._current_observation=current_observation

  def episodic_reset(self, current_observation):
    """
      Reseting internal values. Must be called before each episode.
      Parameters:
        current_observation: Initial observation of this episode.
    """
    self._current_observation = current_observation
    self._machine_stack = []
    self._cumulative_reward = 0.
    self._cumulative_discount = 1.
    self._previous_choice_point = None
    self._tmp_return = None
    
  def _choice_point_handler(self, choice, done=False):
    """
      Record choices and create transitions when possible.
      Parameters:
        choice: A choice, return of the learnable_choice_machine.
        done: Whether this handler is being called on the episode termination or not.
    """
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
      Parameters:
        name: Machine name to be registered
        machine: A machine created by self._create_*_machine
        representation: A representation of this machine. Used when pushing to machine stack. A unique integer is used as a default value.
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
      A convinent decorator for creating and registering functional machine.
      Parameters:
        func: A python function that takes exactly two arguments, HAM and an arbitrary argument.
        representation: A representation of this machine. Used when pushing to machine stack. A unique integer is used as a default value.
    """
    machine_name = func.__name__
    machine = self._create_functional_machine(func)
    self._add_machine(machine_name, machine, representation)
    return func

  def action_machine(self, action_selector: Callable[[Type[HAM], Any], Any], representation=None):
    """
      A convinent decorator for creating and registering action machine
      Parameters:
        action_selector: A python function that takes exactly two arguments, HAM and an arbitrary argument. The return must be an action callable by `action_executor`.
        representation: A representation of this machine. Used when pushing to machine stack. A unique integer is used as a default value.
    """
    machine_name = action_selector.__name__
    machine = self._create_action_machine(action_selector)
    self._add_machine(machine_name, machine, representation)
    return action_selector

  def learnable_choice_machine(self, selector: Callable[[Type[HAM], Any], Any], representation=None):
    """
      A convinent decorator for creating and registering learnable choice machine
      Parameters:
        selector: A python function that takes exactly two arguments, HAM and an arbitrary argument. The return must be a choice, in any form.
        representation: A representation of this machine. Used when pushing to machine stack. A unique integer is used as a default value.
    """
    machine_name = selector.__name__
    machine = self._create_learnable_choice_machine(selector)
    self._add_machine(machine_name, machine, representation)
    return selector

  def _create_functional_machine(self, func: Callable[[Type[HAM], Any], Any]):
    """
      Parameters:
        func: A python function that takes exactly two arguments, HAM and an arbitrary argument.
    """
    if self.verbose:
      print("Functional machine initiated")
    while True:
      args = yield
      ret = func(self, args)
      self._tmp_return = ret

  def _create_action_machine(self, action_selector: Callable[[Type[HAM], Any], Any]):
    """
      Parameters:
        action_selector: A python function that takes exactly two arguments, HAM and an arbitrary argument. The return must be an action callable by `action_executor`.
    """
    if self.verbose:
      print("Action machine initiated")
    while True:
      args = yield
      action = action_selector(self,args)
      self.CALL_action(action)

  def _create_learnable_choice_machine(self, selector: Callable[[Type[HAM], Any], Any]):
    """
      Parameters:
        selector: A python function that takes exactly two arguments, HAM and an arbitrary argument. The return must be a choice, in any form.
    """
    if self.verbose:
      print("Learnable Choice machine initiated")
    while True:
      args = yield
      choice = selector(self, args)
      self._choice_point_handler(choice)
      self._tmp_return = choice

  def CALL(self, machine: Union[str, Callable], args=None):
    """
      A method to CALL a registered machine. This must be used instead of normal python's function calling.
      Parameters:
        machine: Machine's original function or it's name.
        args: An argument to pass into the calling machine. Use compositional data types, list or dictionary, to pass more data.
    """
    if isinstance(machine, str):
      machine_name = machine
    else:
      machine_name = machine.__name__
    assert(machine_name in self.machines)
    try:
      self._machine_stack.append(self.machines[machine_name]["representation"])
      self.machines[machine_name]["machine"].send(args)
      self._machine_stack.pop()
    except StopIteration:
      print("STOP ITERATION??")
      self._tmp_return = None
    return self._tmp_return

  def CALL_action(self, action):
    """
      A convenient method to CALL a trivial action.
      Parameters:
        action: An action to be executed with `action_executor`
    """
    obsv, reward, done, info = self.action_executor(action)
    self.set_observation(obsv)
    self._cumulative_reward+=reward*self._cumulative_discount
    self._cumulative_discount*=self.reward_discount
    if done:
      self._choice_point_handler(None, done = True)
    self._tmp_return = obsv



  
