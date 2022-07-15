from __future__ import annotations # for forward reference of HAM and type hint
import copy
from typing import Tuple, Type, Any, Callable, Union
import threading
from .utils import JointState
import logging
import traceback

# TODO What if choice machines have different kind of choice returned?
# TODO HAMQ-INT internal transition skip?
# TODO Make a machine class instead of dictionary.
# TODO handle ham end (no more choice)

class AlternateLock():
    def __init__(self, initial_thread):
        self.allow = initial_thread
        self.cond = threading.Condition()
        
    def acquire_for(self, thread):
        self.cond.acquire()
        while self.allow!=thread:
            self.cond.wait()
    
    def release_to(self, thread):
        self.allow=thread
        self.cond.notifyAll()
        self.cond.release()


class HAM:

  def __init__(
    self, 
    action_executor: Callable[[Any], Tuple],
    reward_discount: float = 0.9,
    verbose: bool = False
  ):
    """
      A hierarchical of abstract machine (HAM) class.
      Parameters:
        action_executor: A function that handle action execution. Must return (obsv, reward, done, info). Can be as simple as gym's env.step.
    """
    self.action_executor = action_executor
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
    self._tau = 0
    self._tmp_return = None
    self._env_done=False
    try:
      self.terminate()
      self.choice_point_lock.release_to("main")
    except:
      pass
    self.choice_point_lock=None
    self.is_alive = False

  def _choice_point_handler(self, done=False):
    """
      Record choices and create transitions when possible.
      Parameters:
        choice: A choice, return of the learnable_choice_machine.
        done: Whether this handler is being called on the episode termination or not.
    """
    joint_state = JointState(
      s=self._current_observation,
      m=copy.deepcopy(self._machine_stack),
      tau = self._tau
    )
    reward = self._cumulative_reward
    done = 1 if done else 0
    self._cumulative_reward=0.
    self._cumulative_discount = 1.
    self._tau=0
    return joint_state, reward, done, {}

  # def _add_machine(self, name, machine, representation=None):
  #   """
  #     Registering machine created by self._create_*_machine
  #     Parameters:
  #       name: Machine name to be registered
  #       machine: A machine created by self._create_*_machine
  #       representation: A representation of this machine. Used when pushing to machine stack. A unique integer is used as a default value.
  #   """
  #   if self.verbose:
  #     print(name+": \r")
  #   next(machine)
  #   if representation is None:
  #     representation = self.machine_count
  #   self.machine_count+=1
  #   self.machines[name]={
  #     "machine": machine,
  #     "representation": representation,
  #   }

  def machine(self, representation=None):
    def register_func(func, representation=representation):
      if representation is None:
        representation = self.machine_count
      self.machine_count+=1
      machine_name = func.__name__
      self.machines[machine_name]={
        "func": func,
        "representation": representation,
      }
      return func
    return register_func

  # def functional_machine(self, func: Callable[[Type[HAM], Any], Any], representation=None):
  #   """
  #     A convinent decorator for creating and registering functional machine.
  #     Parameters:
  #       func: A python function that takes exactly two arguments, HAM and an arbitrary argument.
  #       representation: A representation of this machine. Used when pushing to machine stack. A unique integer is used as a default value.
  #   """
  #   machine_name = func.__name__
  #   machine = self._create_functional_machine(func)
  #   self._add_machine(machine_name, machine, representation)
  #   return func

  # def action_machine(self, action_selector: Callable[[Type[HAM], Any], Any], representation=None):
  #   """
  #     A convinent decorator for creating and registering action machine
  #     Parameters:
  #       action_selector: A python function that takes exactly two arguments, HAM and an arbitrary argument. The return must be an action callable by `action_executor`.
  #       representation: A representation of this machine. Used when pushing to machine stack. A unique integer is used as a default value.
  #   """
  #   machine_name = action_selector.__name__
  #   machine = self._create_action_machine(action_selector)
  #   self._add_machine(machine_name, machine, representation)
  #   return action_selector

  # def choice_machine(self, machine_name, representation=None):
  #   machine = self._create_choice_machine(machine_name)
  #   self._add_machine(machine_name, machine, representation)
  #   return machine_name

  # def _create_functional_machine(self, func: Callable[[Type[HAM], Any], Any]):
  #   """
  #     Parameters:
  #       func: A python function that takes exactly two arguments, HAM and an arbitrary argument.
  #   """
  #   if self.verbose:
  #     print("Functional machine initiated")
  #   while True:
  #     args = yield
  #     ret = func(self, args)
  #     self._tmp_return = ret

  # def _create_action_machine(self, action_selector: Callable[[Type[HAM], Any], Any]):
  #   """
  #     Parameters:
  #       action_selector: A python function that takes exactly two arguments, HAM and an arbitrary argument. The return must be an action callable by `action_executor`.
  #   """
  #   if self.verbose:
  #     print("Action machine initiated")
  #   while True:
  #     args = yield
  #     action = action_selector(self,args)
  #     self.CALL_action(action)

  # def _create_choice_machine(self, machine_name):
  #   """
  #     Parameters:
  #       selector: A python function that takes exactly two arguments, HAM and an arbitrary argument. The return must be a choice, in any form.
  #   """
  #   if self.verbose:
  #     print("Choice machine initiated")
  #   while True:
  #     args = yield
  #     self.current_choice_machine = machine_name
  #     self.choice_point_lock.release_to("main")
  #     self.choice_point_lock.acquire_for("ham")
  #     # choice = selector(self, args)
  #     # self._choice_point_handler(choice)
  #     # self._tmp_return = choice

  def CALL(self, machine: Union[str, Callable], args=None):
    """
      A method to CALL a registered machine. This must be used instead of normal python's function calling.
      Parameters:
        machine: Machine's original function or it's function name.
        args: An argument to pass into the calling machine. Use compositional data types, list or dictionary, to pass more data.
    """
    if not self.is_alive :
      return 0
    if isinstance(machine, str):
      machine_name = machine
    else:
      machine_name = machine.__name__
    assert(machine_name in self.machines)
    self._machine_stack.append(self.machines[machine_name]["representation"])
    machine_return = None
    try:
      machine_return = self.machines[machine_name]["func"](self, args)
    except Exception as e:
      logging.error(f"{machine_name} machine failed: \n{traceback.format_exc()}\n{str(e)}")
    self._machine_stack.pop()
    return machine_return

  def CALL_action(self, action):
    """
      Parameters:
        action: An action to be executed with `action_executor`
    """
    if not self.is_alive :
      return 0

    try:
      obsv, reward, done, info = self.action_executor(action)
    except Exception as e:
      logging.error(f"Error executing action: \n{traceback.format_exc()}\n{str(e)}")
      return 0
    self._machine_stack.pop()
    self.set_observation(obsv)
    self._cumulative_reward+=reward*self._cumulative_discount
    self._cumulative_discount*=self.reward_discount
    self._tau+=1
    if done:
      self._env_done=True
      self.choice_point_lock.release_to("main") # will send back to termination check in step()
      self.choice_point_lock.acquire_for("ham")
    return obsv, reward, done, info

  def CALL_choice(self, choice_point_name):
    if not self.is_alive :
      return 0
    self.current_choice_machine = choice_point_name
    self.choice_point_lock.release_to("main")
    self.choice_point_lock.acquire_for("ham")
    return self._choice

  def _start(self, machine, args):
      self.choice_point_lock.acquire_for("ham")
      ham_return = self.CALL(machine, args)
      self.choice_point_lock.release_to("main")
      return ham_return

  def start(self, machine: Union[str, Callable], args=None):
    if self.is_alive:
      print("HAM is already running")
      return 0
    
    self.choice_point_lock = AlternateLock("main")
    self.ham_thread = threading.Thread(target = self._start, args=(machine, args))
    self.choice_point_lock.acquire_for("main")
    self.ham_thread.start()
    self.is_alive=True
    print("Starting ham")
    self.choice_point_lock.release_to("ham")
    self.choice_point_lock.acquire_for("main")
    return self._choice_point_handler(done=self._env_done)

  def step(self, choice):
    if not self.is_alive:
      print("HAM is not running. Try reset and start ham")
      return None
    self._choice = choice
    self.choice_point_lock.release_to("ham")
    self.choice_point_lock.acquire_for("main")
    print("step: main acquired")
    machine_state = self._choice_point_handler(done=self._env_done)
    if self._env_done:
      print("Environment terminated")
      self.terminate()
    return machine_state

  def terminate(self):
    if not self.is_alive:
      return None
      
    self.is_alive=False
    self.choice_point_lock.release_to("ham")
    print("released to ham")
    self.ham_thread.join()
    self.choice_point_lock.acquire_for("main")

    self.ham_thread=None
    try:
      self.choice_point_lock.release_to("main")
    except:
      pass
    self.choice_point_lock=None




  
