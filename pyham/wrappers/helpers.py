from tokenize import Single
from typing import Union, Callable, Any
import numpy as np
import gym
from gym import spaces

from ..ham import HAM
from ..utils import JointState
from .single_choice import SingleChoiceTypeEnv
from .multi_choice import MultiChoiceTypeEnv

def create_concat_joint_state_env(ham: HAM, 
                                  env: gym.Env,
                                  initial_machine: Union[Callable, str],
                                  np_pad_config: dict,
                                  initial_args: list = [],
                                  machine_stack_cap: int = 1,
                                  dtype=np.float32,
                                  eval: bool=False,
                                  will_render=False
                                  ):
  """
    A helper function to easily create InducedMDP for HAM `ham` and environment `env`.
    Depends on the number of choicepoints registered in `ham`, `SingleChoiceTypeEnv` or `MultiChoiceTypeEnv` will be instantiated with joint state being the concatenation of `env`'s observation vector and flattened `ham`'s machine stack.

    **
    Currently support only env with Box or MultiDiscrete observation space.
    Machine representations must also be fixed length binary arrays.
    **

    Parameters:
      ham: HAM
      env: gym.Env
      np_pad_config: dictionary containing numpy.pad's mode and the reset of kwargs.
        e.g. 
        np_pad_config = {
          "mode": "constant", # (default)
          "constant_values": 0
        }
      machine_stack_cap: Number of top most machines to include in the joint state representation, pad if needed.
      dtype: data type for `spaces.Box`
      eval: whether to instantiate env for evaluation or not.
      will_render: If true, pre-render every frames even if `render()` is not being called. Must be set to true if `render()` method is expected to be called.
    Return:
      Wrapped env with joint state representation defined as concatenated numpy array between original env's observation space and fixed length of machine stack representations.
  """

  if len(ham.cpm)<1:
    raise Exception("Unable to create env from HAM without choicepoint.")

  if len(ham.cpm)==1:
    wrapped_env = SingleChoiceTypeEnv
  else:
    wrapped_env = MultiChoiceTypeEnv

  def _obj_len(obj):
    if hasattr(obj, '__len__'):
      return len(obj)
    else:
      return 1

  repr_length = None
  for machine_name in ham.machines:
    repr = ham.get_machine_repr(machine_name)
    if repr_length is None:
      repr_length = _obj_len(repr)
    elif repr_length != _obj_len(repr):
      raise Exception("Unable to create joint state representation due to inconsistent representation length.")
  
  build_js_args = (env.observation_space, repr_length, np_pad_config, machine_stack_cap)
  if isinstance(env.observation_space, spaces.Box):
    js_space, js2repr = _concat_Box_joint_state(*build_js_args, dtype)
  elif isinstance(env.observation_space, spaces.MultiDiscrete):
    js_space, js2repr = _concat_MultiDiscrete_joint_state(*build_js_args)
  else:
    raise Exception("Unsupported observation space type.")
  return wrapped_env(ham, 
                    env,
                    js_space, 
                    js2repr, 
                    initial_machine=initial_machine,
                    initial_args=initial_args,
                    eval=eval,
                    will_render=will_render)

def _concat_Box_joint_state(og_space, repr_length, np_pad_config: dict, machine_stack_cap: int, dtype):
  machine_stack_repr_shape = (machine_stack_cap*repr_length,)
  machine_stack_high = np.ones(machine_stack_repr_shape[0])
  machine_stack_low = np.zeros(machine_stack_repr_shape[0])
  og_obsv_high = og_space.high
  og_obsv_low = og_space.low
  js_space = spaces.Box(np.hstack((og_obsv_low, machine_stack_low)),
                                np.hstack((og_obsv_high, machine_stack_high)),
                                dtype = dtype)
  def js2repr(js: JointState):
    if len(js.m)==0:
      machine_stack_repr = np.array([])
    else:
      machine_stack_repr = np.hstack(js.m[-machine_stack_cap:])
    padding = max(0, machine_stack_repr_shape[0]-len(machine_stack_repr))
    machine_stack_repr = np.pad(machine_stack_repr, (padding,0), **np_pad_config)
    js_repr = np.float32(np.hstack((js.s,machine_stack_repr)))
    return js_repr

  return js_space, js2repr
  
def _concat_MultiDiscrete_joint_state(og_space, repr_length, np_pad_config: dict, machine_stack_cap: int):
  og_nvec = og_space.nvec
  machine_stack_repr_shape = (machine_stack_cap*repr_length,)
  machine_stack_nvec = np.int64(np.ones(machine_stack_repr_shape)*2)
  js_space = spaces.MultiDiscrete(np.hstack((og_nvec, machine_stack_nvec)))

  def js2repr(js: JointState):
    if len(js.m)==0:
      machine_stack_repr = np.array([])
    else:
      machine_stack_repr = np.hstack(js.m[-machine_stack_cap:])
    padding = max(0, machine_stack_repr_shape[0]-len(machine_stack_repr))
    machine_stack_repr = np.pad(machine_stack_repr, (padding,0), **np_pad_config)
    js_repr =np.int64(np.hstack((js.s,machine_stack_repr)))
    return js_repr
  return js_space, js2repr