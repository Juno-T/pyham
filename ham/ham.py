from __future__ import annotations # for forward reference of HAM and type hint
import copy
from typing import Tuple, Type, Any, Callable, Union
import logging
import traceback
import threading

from .utils import JointState, AlternateLock

# TODO What if choice machines have different kind of choice returned?
# TODO HAMQ-INT internal transition skip?

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
    self.current_choice_point_name=None
    self._machine_stack = []
    self._cumulative_reward = 0.
    self._cumulative_discount = 1.
    self._tau = 0
    self._tmp_return = None
    self._env_done=False
    try:
      self.terminate()
      self._choice_point_lock.release_to("main")
    except:
      pass
    self._choice_point_lock=None
    self.is_alive = False

  def get_info(self):
    return {
      "next_choice_point": self.current_choice_point_name
    }

  def _choice_point_handler(self, done=False):
    """
      Record choices and create transitions when possible.
      Parameters:
        choice: A choice, return of the learnable_choice_machine.
        done: Whether this handler is being called on the episode termination or not.
      Returns:
        A 4 items tuple:
          joint state:
          cumulative reward:
          done: Environment done or ham done.
          info: dictionary with extra info
    """
    joint_state = JointState(
      s=self._current_observation,
      m=copy.deepcopy(self._machine_stack),
      tau = self._tau
    )
    reward = self._cumulative_reward
    done = 1 if done or (not self.is_alive) else 0
    self._cumulative_reward=0.
    self._cumulative_discount = 1.
    self._tau=0
    return joint_state, reward, done, self.get_info()

  def machine(self, func):
    """
      convinent decorator for defining macchine without representation
    """
    self.machine_with_repr()(func)
    return func

  def machine_with_repr(self, representation=None):
    if representation is None:
      representation = self.machine_count
    self.machine_count+=1
    
    def register_func(func: Callable[[Type(HAM),],Any], representation=representation):
      machine_name = func.__name__
      self.machines[machine_name]={
        "func": func,
        "representation": representation,
      }
      return func
    return register_func

  def CALL(self, machine: Union[str, Callable], args=[]):
    """
      A method to CALL a registered machine. This must be used instead of normal python's function calling.
      Parameters:
        machine: Machine's original function or it's function name.
        args: An argument to pass into the calling machine. Must be list or tuple.
    """
    if not self.is_alive :
      return 0
    if isinstance(machine, str):
      machine_name = machine
    else:
      machine_name = machine.__name__
    if not machine_name in self.machines:
      logging.error(f"{machine_name} machine not registered")
      return None
    assert(machine_name in self.machines)
    self._machine_stack.append(self.machines[machine_name]["representation"])
    machine_return = None
    try:
      machine_return = self.machines[machine_name]["func"](self, *args)
    except Exception as e:
      self._machine_stack.pop()
      logging.error(f"{machine_name} machine failed: \n{traceback.format_exc()}\n{str(e)}")
      return 0
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
    self.set_observation(obsv)
    self._cumulative_reward+=reward*self._cumulative_discount
    self._cumulative_discount*=self.reward_discount
    self._tau+=1
    if done:
      self._env_done=True
      self._choice_point_lock.release_to("main") # will send back to termination check in step()
      self._choice_point_lock.acquire_for("ham")
    return obsv, reward, done, info

  def CALL_choice(self, choice_point_name):
    if not self.is_alive :
      return 0
    self.current_choice_point_name = choice_point_name
    self._choice_point_lock.release_to("main")
    self._choice_point_lock.acquire_for("ham")
    return self._choice

  def _start(self, machine, args):
      self._choice_point_lock.acquire_for("ham")
      ham_return = self.CALL(machine, args)
      self.is_alive=False
      self._choice_point_lock.release_to("main")
      return ham_return

  def start(self, machine: Union[str, Callable], args=[]):
    if self.is_alive:
      logging.warning("HAM is already running")
      return 0
    
    self._choice_point_lock = AlternateLock("main")
    self.ham_thread = threading.Thread(target = self._start, args=(machine, args))
    self._choice_point_lock.acquire_for("main")
    self.ham_thread.start()
    self.is_alive=True
    self._choice_point_lock.release_to("ham")
    self._choice_point_lock.acquire_for("main")
    return self._choice_point_handler(done=self._env_done)

  def step(self, choice):
    if not self.is_alive:
      logging.warning("HAM is not running. Try restart ham")
      return None
    self._choice = choice
    self._choice_point_lock.release_to("ham")
    self._choice_point_lock.acquire_for("main")
    machine_state = self._choice_point_handler(done=self._env_done)
    if self._env_done:
      self.terminate()
    return machine_state

  def terminate(self):
    if not self.is_alive:
      return None
      
    self.is_alive=False
    self._choice_point_lock.release_to("ham")
    self.ham_thread.join()
    self._choice_point_lock.acquire_for("main")

    self.ham_thread=None
    try:
      self._choice_point_lock.release_to("main")
    except:
      pass
    self._choice_point_lock=None




  
