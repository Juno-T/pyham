from typing import Union, Callable, Any

import numpy as np


def onehot_repr(machine_id: int, total_machine: int)->np.ndarray:
  repr = np.zeros((total_machine))
  repr[machine_id]=1
  return repr


class MachineRepresentation:

  predefined_representation = {
    "onehot": onehot_repr
  }

  def __init__(self, representation: Union[str, Callable[[int, int], Any]]):
    self.repr_func = None
    if isinstance(representation, str):
      assert representation in self.predefined_representation, \
        f"Representation {representation} mode is undefined. \n\
          Must be one of {list(self.predefined_representation.keys())}"
      self.repr_func = self.predefined_representation[representation]
    elif isinstance(representation, Callable):
      self.repr_func = representation

    self._cache = []
    self.total_machine = 0

  def reset(self, total_machine: int):
    self._cache = [None]*total_machine
    self.total_machine = total_machine

  def get_repr(self, machine_id: int):
    if self._cache[machine_id] is None:
      self._cache[machine_id] = self.repr_func(machine_id, self.total_machine)
    return self._cache[machine_id]
