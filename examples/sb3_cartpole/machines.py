import numpy as np
from gym import spaces

from pyham.ham import HAM

def create_trivial_cartpole_ham(discount):
  trivial_cartpole_ham = HAM(discount, representation="onehot")

  @trivial_cartpole_ham.machine
  def top_loop(ham):
    while ham.is_alive:
      action = int(ham.CALL_choice("binary action"))
      ham.CALL_action(action)
  
  choice_space = spaces.Discrete(2)

  return trivial_cartpole_ham, choice_space, top_loop, []

def create_balance_recover_cartpole_ham(discount):
  cartpole_ham = HAM(discount, representation="onehot")

  @cartpole_ham.machine
  def top_loop(ham):
    while ham.is_alive:
      choice = int(ham.CALL_choice("balance-recover")) # 3 choices: 0 & 1 are recovering, 2 is balancing
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

  choice_space = spaces.Discrete(3)
  return cartpole_ham, choice_space, top_loop, []