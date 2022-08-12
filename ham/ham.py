from __future__ import annotations # for forward reference of HAM and type hint
import copy
from gym import spaces
from typing import Tuple, Type, Any, Callable, Union, Optional
import logging
import traceback
import threading

from .utils import JointState, AlternateLock, MachineRepresentation
from .choicepoint import Choicepoint, ChoicepointsManager

# TODO What if choice machines have different kind of choice returned?
# TODO HAMQ-INT internal transition skip?

class HAM:

  def __init__(
    self, 
    action_executor: Optional[Callable[[Any], Tuple]] = None,
    representation: Optional[Union[str, Callable[[int, int], Any]]] = "onehot",
  ):
    """
      A hierarchical of abstract machine (HAM) class.
      Parameters:
        action_executor: A function that handle action execution. Must return (obsv, reward, done, info). Can be as simple as gym's env.step.
    """
    self.action_executor = action_executor
    self.machine_repr = MachineRepresentation(representation)
    self.machines = {}
    self.machine_count=0
    self.cpm = ChoicepointsManager(eval=False) # TODO
    self._is_alive=False

  @property
  def is_alive(self):
    return self._is_alive

  @property
  def current_observation(self):
    return self._current_observation

  def get_machine_repr(self, machine_name):
    """
      Getting machine representation by name
      Parameters:
        machine_name: machine's name
      Return
        machine_representation calculated according to self.machine_repr
    """
    assert(machine_name in self.machines)
    repr = self.machines[machine_name]["representation"]
    if repr is None:
      repr = self.machine_repr.get_repr(self.machines[machine_name]["id"])
    return repr

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
        current_observation: Initial observation of this episode. i.e. a return from `gym.env.reset()`
    """
    try:
      self.terminate()
      self._choice_point_lock.release_to("main")
    except:
      pass
    self._choice_point_lock=None
    self._is_alive = False

    self._current_observation = current_observation
    self.current_choice_point_name=None
    self._machine_stack = []
    self._cumulative_actual_reward = 0.
    self._tau = 0
    self._tmp_return = None
    self._env_done=False
    

  def get_info(self):
    """
      Information to put in info which will be return at each checkpoint
    """
    return {
      "next_choice_point": self.current_choice_point_name,
      "actual_reward": self._cumulative_actual_reward,
    }

  def _choice_point_handler(self, done=False):
    """
      Handling the return tuple at choice point.
      Parameters:
        choice: A choice, return of the learnable_choice_machine.
        done: Whether this handler is being called on the episode termination or not.
      Returns:
        A 4 items tuple:
          joint state:  `JointState` of env state and machine stack at choice point.
          cumulative reward: Cumulative reward
          done: Environment done or ham done.
          info: dictionary with extra info, e.g. info['next_choice_point']
    """
    cp_name = self.current_choice_point_name
    cp_reward, cp_tau = self.cpm.reset_choicepoint(cp_name)
    joint_state = JointState(
      s=self._current_observation,
      m=copy.deepcopy(self._machine_stack),
      tau = cp_tau
    )
    reward = cp_reward
    # if self.eval: # TODO
    done = done or (not self._is_alive)
    info = self.get_info()
    # self._cumulative_actual_reward = 0.
    # self._tau=0
    return joint_state, reward, bool(done), info

  def machine(self, func: Callable[[Type(HAM),],Any]):
    """
      A convinent decorator for registering a machine with default machine representation.
      Parameters:
        func: A python function to be registered. 
    """
    self.machine_with_repr()(func)
    self.machine_repr.reset(self.machine_count)
    return func

  def machine_with_repr(self, representation=None):
    """
      A decorator to register a machine with representation.
      Parameters:
        representation: A representation of the machine to be registered
        decorated function: A python function to be registered.
    """
    id = self.machine_count
    self.machine_count+=1
    
    def register_func(func: Callable[[Type(HAM),],Any], id=id, representation=representation):
      machine_name = func.__name__
      self.machines[machine_name]={
        "func": func,
        "id": id,
        "representation": representation,
      }
      return func
    return register_func

  def choicepoint(self, name: str, choice_space: spaces.Space, discount: float):
    """
      Define choicepoint
    """
    if name in self.cpm.choicepoints_order:
      logging.warn(f"Choice point named {name} is already existed. Ignore new assignment.")
      return 
    choicepoint = Choicepoint(name, choice_space, discount)
    self.cpm.add_choicepoint(choicepoint)
    return choicepoint
    
  def CALL(self, machine: Union[str, Callable], args=[]):
    """
      A method to CALL a registered machine. This must be used instead of normal python's function calling.
      Parameters:
        machine: Machine's original function or it's function name.
        args: An argument to pass into the calling machine. Must be a list or tuple.
    """
    if not self._is_alive :
      return 0

    if isinstance(machine, str):
      machine_name = machine
    else:
      machine_name = machine.__name__

    if not machine_name in self.machines:
      logging.error(f"{machine_name} machine not registered")
      return None

    if (not isinstance(args, list)) and (not isinstance(args, tuple)):
      logging.error(f"Argument {args} must be list or tuple, not {type(args)}")
      return None
    
    assert(machine_name in self.machines)
    m_repr = self.get_machine_repr(machine_name)
    self._machine_stack.append(m_repr)
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
      A function to interact with environment given an action.
      Parameters:
        action: An action to be executed with `action_executor`
    """
    if not self._is_alive :
      return 0

    try:
      obsv, reward, done, info = self.action_executor(action)
    except Exception as e:
      logging.error(f"Error executing action: \n{traceback.format_exc()}\n{str(e)}")
      return 0
    self.set_observation(obsv)
    self.cpm.distribute_reward(reward)
    self._cumulative_actual_reward += reward
    self._tau+=1
    if done:
      self._env_done=True
      self._choice_point_lock.release_to("main") # will send back to termination check in step()
      self._choice_point_lock.acquire_for("ham")
    return obsv, reward, done, info

  def CALL_choice(self, choicepoint: Union[str, Choicepoint]):
    """
      A function to make a choice point in HAMs.
      Parameters:
        choicepoint: choice point or it's name.
      Return:
        choice: A choice selected by HAM.step()
    """
    if not self._is_alive :
      return 0
    if isinstance(choicepoint, Choicepoint):
      choice_point_name = choicepoint.name
    else:
      choice_point_name = choicepoint
    self.current_choice_point_name = choice_point_name
    self._choice_point_lock.release_to("main")
    self._choice_point_lock.acquire_for("ham")
    if not self._is_alive :
      return 0
    return self._choice

  def _start(self, machine, args):
    """
      HAM thread function.
      Parameters:
        machine: Registered machine or its name.
        args: Argument to the machine
    """
    self._choice_point_lock.acquire_for("ham")
    ham_return = self.CALL(machine, args)
    self._is_alive=False
    self._choice_point_lock.release_to("main")
    return ham_return

  def start(self, machine: Union[str, Callable], args=[]):
    """
      Start HAMs from specified machine.
      Parameters:
        machine: A registered machine or its name. Should be the highest level machine.
        args: Arguments for the machine. Must be list or tuple
    """
    if self._is_alive:
      logging.warning("HAM is already running")
      return 0
    
    if self.action_executor is None:
      raise("`action_executor` must be defined.")
    
    if (not isinstance(args, list)) and (not isinstance(args, tuple)):
      raise(f"Argument {args} must be list or tuple, not {type(args)}")
    
    self._choice_point_lock = AlternateLock("main")
    self.cpm.reset()
    self.ham_thread = threading.Thread(target = self._start, args=(machine, args))
    self._choice_point_lock.acquire_for("main")
    self.ham_thread.start()
    self._is_alive=True
    self._choice_point_lock.release_to("ham")
    self._choice_point_lock.acquire_for("main")
    return self._choice_point_handler(done=self._env_done)

  def step(self, choice):
    """
      Iterate the running HAMs giving choice at the choice point.
      Paremters:
        choice: A choice to be used by the running HAMs.
      Return:
        A 4 items tuple:
          joint state:  `JointState` of env state and machine stack at choice point.
          cumulative reward: Cumulative reward
          done: Environment done or ham done.
          info: dictionary with extra info, e.g. info['next_choice_point']
    """
    if not self._is_alive:
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
    """
      Force terminate the HAMs if running.
    """
    if not self._is_alive:
      return None
      
    self._is_alive=False
    self._choice_point_lock.release_to("ham")
    self.ham_thread.join()
    self._choice_point_lock.acquire_for("main")

    self.ham_thread=None
    try:
      self._choice_point_lock.release_to("main")
    except:
      pass
    self._choice_point_lock=None




  
