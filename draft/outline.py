from __future__ import annotations # for forward reference of HAM and type hint
from typing import Tuple, Type, Any, Callable

class HAM:
  def __init__(self, action_executor: Callable[[Any], Tuple]):
    self.action_executor = action_executor
    self.current_observation = None
    self.current_machine = None
    self.machines = {}
    self.state_stack = []
    self._tmp_return = None

  def set_observation(self, obsv):
    self.current_observation=obsv

  def set_machine(self, machine):
    self.current_machine = machine

  def add_machine(self, name, machine):
    print(name+": \r")
    next(machine)
    self.machines[name]=machine

  def create_functional_machine(self, func: Callable[[Type[HAM], Any], None]):
    """
      Parameters:
        func: a function
    """
    print("Functional machine initiated")
    while True:
      smth = yield # smth should be unified
      ret = func(self, smth)
      self._tmp_return = ret

  def create_action_machine(self, action_selector):
    print("Action machine initiated")
    while True:
      smth = yield # smth should be unified
      action = action_selector(self,smth)
      obsv, reward, done, info = self.action_executor(action)
      self.set_observation(obsv)
      self._tmp_return = obsv

  def create_learnable_choice_machine(self, selector):
    """
      Need something to record choice for transitions
    """
    print("Learnable Choice machine initiated")
    while True:
      smth = yield
      choice = selector(self, smth)
      self._tmp_return = choice

  def CALL(self, machine_name):
    assert(machine_name in self.machines)
    try:
      self.state_stack.append(machine_name)
      self.machines[machine_name].send(self.current_observation) # TODO this will define all the smth
      self.state_stack.pop()
    except StopIteration:
      print("STOP ITERATION??")
      self._tmp_return = None
    return self._tmp_return


  
