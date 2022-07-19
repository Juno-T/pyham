import numpy as np
from gym import spaces

from pyham.ham import HAM

def create_trivial_cartpole_ham(discount):
  trivial_cartpole_ham = HAM(discount)

  num_machines = 1
  reprs = np.eye(num_machines)

  @trivial_cartpole_ham.machine_with_repr(reprs[0])
  def top_loop(ham):
    while ham.is_alive:
      action = int(ham.CALL_choice("binary action"))
      ham.CALL_action(action)
  
  choice_space = spaces.Discrete(2)

  return trivial_cartpole_ham, choice_space, top_loop, []