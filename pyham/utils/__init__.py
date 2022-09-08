import collections
import logging

from .units import JointState, Transition
from .threading import AlternateLock
from .machine_representation import MachineRepresentation

def deprecated(alternative: str=""):
  def add_warning(func, alternative=alternative):
    def deprecated_func(*args, **kwargs):
      if alternative=="":
        logging.warn(f"\nDeprecated: Function {func.__name__} will be removed.")
      else:
        logging.warn(f"\nDeprecated: Please use {alternative} instead of {func.__name__}.")
      return func(*args, **kwargs)
    return deprecated_func
  return add_warning

def deep_dict_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_dict_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source