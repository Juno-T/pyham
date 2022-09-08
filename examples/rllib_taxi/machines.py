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
    Creating a Get-Put ham with two choice points for taxi environment.
    Here we use default setting (`representation="onehot"`) when instantiating HAM which is equivalent of manually assignment in /examples/sb3_taxi/machine.py .
  """
  taxi_ham = HAM(representation="onehot")

  get_put = taxi_ham.choicepoint("get_put", spaces.Discrete(2), discount=1, discount_correction=1/0.99)
  nav = taxi_ham.choicepoint("nav", spaces.Discrete(5), discount=1, discount_correction=1/0.99)

  @taxi_ham.machine
  def root(ham: HAM):
    while ham.is_alive:
      m = ham.CALL_choice(get_put)
      if m==0:
        ham.CALL(get)
      else:
        ham.CALL(put)

  @taxi_ham.machine
  def get(ham: HAM):
    ham.CALL(navigate) # navigate
    ham.CALL_action(4) # try pick up

  @taxi_ham.machine
  def put(ham: HAM):
    ham.CALL(navigate) # navigate
    ham.CALL_action(5) # try drop off
    if ham.is_alive: # If drop on the wrong destination
      ham.CALL_action(4) # pick up and try again

  @taxi_ham.machine
  def navigate(ham: HAM):
    while ham.is_alive:
      direction = int(ham.CALL_choice(nav)) # 5 choices: SNEW & noop
      if direction==4:
        break
      ham.CALL_action(direction)

  return taxi_ham, root, []