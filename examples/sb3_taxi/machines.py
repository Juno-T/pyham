import numpy as np
from gym import spaces

from pyham.ham import HAM

def create_trivial_taxi_ham(discount):
  trivial_taxi_ham = HAM(discount)

  num_machines = 1
  reprs = np.eye(num_machines)

  @trivial_taxi_ham.machine_with_repr(reprs[0])
  def top_loop(ham):
    while ham.is_alive:
      action = int(ham.CALL_choice("taxi action")) # 6 actions
      ham.CALL_action(action)
  
  choice_space = spaces.Discrete(6)

  return trivial_taxi_ham, choice_space, top_loop, []


def create_taxi_ham(discount):
  """
  """
  taxi_ham = HAM(discount)
  num_machines = 4
  reprs = np.eye(num_machines)

  @taxi_ham.machine_with_repr(representation=reprs[0])
  def root(ham: HAM):
    ham.CALL(get)
    ham.CALL(put)

  @taxi_ham.machine_with_repr(representation=reprs[1])
  def get(ham: HAM):
    _, _, psg_pos, _  = ham.current_observation
    while ham.is_alive and int(psg_pos)!=4: # while passenger is not in taxi
      ham.CALL(navigate) # navigate
      ham.CALL_action(4) # try pick up
      _, _, psg_pos, _  = ham.current_observation

  @taxi_ham.machine_with_repr(representation=reprs[2])
  def put(ham: HAM):
    while ham.is_alive: # while not end
      ham.CALL(navigate) # navigate
      ham.CALL_action(5) # try drop off
      if ham.is_alive: # If drop on the wrong destination
        ham.CALL_action(4) # pick up and try again

  @taxi_ham.machine_with_repr(representation=reprs[3])
  def navigate(ham: HAM):
    while ham.is_alive:
      direction = int(ham.CALL_choice("navigate")) # 5 choices: SNEW & noop
      if direction==4:
        break
      ham.CALL_action(direction)

  choice_space = spaces.Discrete(5)

  return taxi_ham, choice_space, root, []