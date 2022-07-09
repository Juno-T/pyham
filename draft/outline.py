from __future__ import annotations # for forward reference of HAM and type hint
from typing import Tuple, Type, Any, Callable, Union

# TODO record choices
# TODO integrate with transitions replay
# TODO Machine's customizable parameters
# TODO refactor

class HAM:
  def __init__(self, action_executor: Callable[[Any], Tuple]):
    """
      Parameters:
        action_executor: A function that handle action execution. Must return (obsv, reward, done, info). Can be as simple as gym's env.step
    """
    self.action_executor = action_executor
    self.current_observation = None
    self.machines = {}
    self.state_stack = []
    self._tmp_return = None

  def set_observation(self, obsv):
    self.current_observation=obsv

  def _add_machine(self, name, machine):
    """
      Registering machine created by self._create_*_machine
    """
    print(name+": \r")
    next(machine)
    self.machines[name]=machine

  def functional_machine(self, func: Callable[[Type[HAM], Any], Any]):
    """
      A convinent decorator for creating and registering functional machine
    """
    machine_name = func.__name__
    machine = self._create_functional_machine(func)
    self._add_machine(machine_name, machine)
    return func

  def action_machine(self, action_selector: Callable[[Type[HAM], Any], Any]):
    """
      A convinent decorator for creating and registering action machine
    """
    machine_name = action_selector.__name__
    machine = self._create_action_machine(action_selector)
    self._add_machine(machine_name, machine)
    return action_selector

  def learnable_choice_machine(self, selector: Callable[[Type[HAM], Any], Any]):
    """
      A convinent decorator for creating and registering learnable choice machine
    """
    machine_name = selector.__name__
    machine = self._create_learnable_choice_machine(selector)
    self._add_machine(machine_name, machine)
    return selector

  def _create_functional_machine(self, func: Callable[[Type[HAM], Any], Any]):
    """
      Parameters:
        func: A pure python function
    """
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
    print("Action machine initiated")
    while True:
      smth = yield # smth should be unified
      action = action_selector(self,smth)
      obsv, reward, done, info = self.action_executor(action)
      self.set_observation(obsv)
      self._tmp_return = obsv

  def _create_learnable_choice_machine(self, selector: Callable[[Type[HAM], Any], Any]):
    """
      Parameters:
        selector: A pure function that return an option given the input
    """
    print("Learnable Choice machine initiated")
    while True:
      smth = yield
      choice = selector(self, smth)
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
      self.state_stack.append(machine_name)
      self.machines[machine_name].send(self.current_observation) # TODO this will define all the smth
      self.state_stack.pop()
    except StopIteration:
      print("STOP ITERATION??")
      self._tmp_return = None
    return self._tmp_return


  
