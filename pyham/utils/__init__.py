from .types import JointState, Transition
from .threading import AlternateLock
from .machine_representation import MachineRepresentation
import logging

def deprecated(alternative: str=""):
  def add_warning(func, alternative=alternative):
    def deprecated_func(*args, **kwargs):
      if alternative=="":
        logging.warn(f"Deprecated: Function {func.__name__} will be removed.")
      else:
        logging.warn(f"Deprecated: Please use {alternative} instead of {func.__name__}.")
      return func(*args, **kwargs)
    return deprecated_func
  return add_warning