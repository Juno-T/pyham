import numpy as np
from gym import spaces

from pyham import HAM

def create_trivial_taxi_ham():
  trivial_taxi_ham = HAM()

  num_machines = 1
  reprs = np.eye(num_machines)

  original_choice = trivial_taxi_ham.choicepoint("original_choice", spaces.Discrete(6), discount=1)
  @trivial_taxi_ham.machine_with_repr(reprs[0])
  def top_loop(ham):
    while ham.is_alive:
      action = int(ham.CALL_choice(original_choice)) # 6 actions
      ham.CALL_action(action)

  return trivial_taxi_ham, top_loop, []


def create_taxi_ham():
  """
    Creating a Get-Put ham for taxi environment.
    Here we manually assigning machine's representation using onehot vector as an example, and it is equivalent of using default setting (`representation="onehot"`) when instantiating HAM without overriding representation during machine registration.
  """
  taxi_ham = HAM()
  num_machines = 4
  reprs = np.eye(num_machines)

  nav = taxi_ham.choicepoint("navigation", spaces.Discrete(5), discount=1)
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
      direction = int(ham.CALL_choice(nav)) # 5 choices: SNEW & noop
      if direction==4:
        break
      ham.CALL_action(direction)

  return taxi_ham, root, []