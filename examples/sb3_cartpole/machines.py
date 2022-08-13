import numpy as np
from gym import spaces

from pyham import HAM

def create_trivial_cartpole_ham(discount):
  trivial_cartpole_ham = HAM(discount, representation="onehot")

  binary_choice = trivial_cartpole_ham.choicepoint("binary_choice", spaces.Discrete(2), discount=1)
  @trivial_cartpole_ham.machine
  def top_loop(ham):
    while ham.is_alive:
      action = int(ham.CALL_choice(binary_choice))
      ham.CALL_action(action)

  return trivial_cartpole_ham, top_loop, []

def create_balance_recover_cartpole_ham(discount):
  cartpole_ham = HAM(discount, representation="onehot")

  balance_recover_choice = cartpole_ham.choicepoint("balance_recover_choice", spaces.Discrete(3), discount=1)
  @cartpole_ham.machine
  def top_loop(ham):
    while ham.is_alive:
      choice = int(ham.CALL_choice(balance_recover_choice)) # 3 choices: 0 & 1 are recovering, 2 is balancing
      if choice==2:
        ham.CALL(balance)
      else:
        ham.CALL(recover, args=(choice,))

  @cartpole_ham.machine
  def balance(ham):
    ham.CALL_action(0)
    ham.CALL_action(1)

  @cartpole_ham.machine
  def recover(ham, action):
    ham.CALL_action(action)
    ham.CALL_action(action)

  return cartpole_ham, top_loop, []