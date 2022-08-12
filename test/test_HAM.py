import unittest
import pytest
import numpy as np
from gym import spaces

from pyham.ham import HAM
from numpy.random import default_rng


class TestFunctionality(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    self.rng = default_rng(42)

    def env_exe(action):
      return "observation", 0.5, action=="end_env", "info"
    self.env_exe = env_exe
    self.myham = HAM(env_exe)
    return super().setUp()

  @pytest.mark.timeout(3)
  def test_initialize(self):
    choicepoint1 = self.myham.choicepoint("choice name", spaces.Discrete(2), 1)
    @self.myham.machine
    def loop_machine(ham: HAM):
      for i in range(10):
        ham.CALL(machine2)

    @self.myham.machine
    def machine2(ham: HAM):
      for i in range(10):
        ham.CALL_choice("choice name")
    self.assertTrue(self.myham.machine_count==2)

  @pytest.mark.timeout(3)
  def test_simple(self):
    m2_choice_point = self.myham.choicepoint("m2_choice_point", spaces.Discrete(2), 1)
    @self.myham.machine
    def loop_machine(ham: HAM):
      for i in range(10):
        ham.CALL(machine2)

    @self.myham.machine
    def machine2(ham: HAM):
      ham.CALL_choice("m2_choice_point")

    self.myham.episodic_reset("initial observation")
    self.assertTrue(self.myham.is_alive == False)
    self.assertTrue(self.myham._choice_point_lock == None)
    self.assertTrue(self.myham._current_observation=="initial observation")

    machine_state, reward, done, info = self.myham.start(loop_machine)
    if not done:
      machine_state, reward, done, info = self.myham.step("my choice")
      self.assertTrue(machine_state.s == "initial observation")
      self.assertTrue(np.array_equal(machine_state.m, np.array([[1,0],[0,1]]))) # machine stack
      self.assertTrue(reward == 0)
      self.assertTrue(done==False)
      self.assertTrue(info["next_choice_point"] == "m2_choice_point")
    self.myham.terminate()
  
  def test_passing_machine_argument(self):
    m2_choice_point = self.myham.choicepoint("m2_choice_point", spaces.Discrete(2), 1)
    @self.myham.machine
    def loop_machine(ham: HAM):
      for i in range(10):
        ham.CALL(machine2, (m2_choice_point,))

    @self.myham.machine
    def machine2(ham: HAM, cp_name):
      ham.CALL_choice(cp_name)

    self.myham.episodic_reset("initial observation")

    machine_state, reward, done, info = self.myham.start(loop_machine)

    self.assertTrue(info['next_choice_point']==m2_choice_point.name)
    for i in range(1,10):
      machine_state, reward, done, info = self.myham.step("my choice")
      self.assertTrue(info["next_choice_point"] == m2_choice_point.name)
    self.myham.terminate()

  def test_machine_repr(self):
    m2_choice_point = self.myham.choicepoint("m2_choice_point", spaces.Discrete(2), 1)
    repr = np.eye(2)
    @self.myham.machine_with_repr(representation=repr[0])
    def loop_machine(ham: HAM):
      for i in range(10):
        ham.CALL(machine2)

    @self.myham.machine_with_repr(representation=repr[1])
    def machine2(ham: HAM):
      ham.CALL_choice("m2_choice_point")

    self.myham.episodic_reset("initial observation")

    machine_state, reward, done, info = self.myham.start(loop_machine)

    m_stack = machine_state.m
    self.assertTrue(np.array_equal(np.stack(m_stack), np.eye(2)))
    for i in range(1,10):
      machine_state, reward, done, info = self.myham.step("my choice")
      m_stack = machine_state.m
      self.assertTrue(np.array_equal(np.stack(m_stack), np.eye(2)))
    self.myham.terminate()
  
  def test_rep_actions_transitions(self):
    repetition_choice = self.myham.choicepoint("repetition", spaces.Discrete(10), 1)
    @self.myham.machine
    def loop_machine(ham: HAM):
      for i in range(10):
        rep = ham.CALL_choice(repetition_choice)
        for _ in range(rep):
          ham.CALL_action("action")

    self.myham.episodic_reset("initial observation")

    machine_state, reward, done, info = self.myham.start(loop_machine)
    self.assertTrue(info['next_choice_point'] == repetition_choice.name)
    for i in range(10):
      machine_state, reward, done, info = self.myham.step(i)
      self.assertTrue(info['next_choice_point'] == repetition_choice.name)
      self.assertTrue(machine_state.tau==i)
      self.assertTrue(reward == 0.5*i)
  
  def test_env_end(self):
    choice = self.myham.choicepoint("choice", spaces.Discrete(10), 1)
    @self.myham.machine
    def loop_machine(ham: HAM):
      while ham.is_alive:
        action = ham.CALL_choice(choice)
        ham.CALL_action(action)

    self.myham.episodic_reset("initial observation")

    machine_state, reward, done, info = self.myham.start(loop_machine)
    self.assertTrue(info['next_choice_point'] == choice.name)
    for i in range(10):
      machine_state, reward, done, info = self.myham.step("action")
    machine_state, reward, done, info = self.myham.step("end_env")
    self.assertTrue(done)
    self.assertTrue(self.myham.is_alive==False)

  def test_ham_end(self):
    choice = self.myham.choicepoint("choice", spaces.Discrete(10), 1)
    @self.myham.machine
    def loop_machine(ham: HAM):
      for i in range(5):
        action = ham.CALL_choice(choice)
        ham.CALL_action(action)

    self.myham.episodic_reset("initial observation")

    machine_state, reward, done, info = self.myham.start(loop_machine)
    self.assertTrue(info['next_choice_point'] == choice.name)
    while True:
      machine_state, reward, done, info = self.myham.step("action")
      if done:
        break
    self.assertTrue(done)
    self.assertTrue(self.myham.is_alive==False)

  def test_callable_repr(self):
    numbering = lambda id, total: id
    myham = HAM(self.env_exe, numbering)

    m2_choice_point = myham.choicepoint("m2_choice_point", spaces.Discrete(2), 1)
    @myham.machine
    def loop_machine(ham: HAM):
      for i in range(10):
        ham.CALL(machine2)

    @myham.machine
    def machine2(ham: HAM):
      ham.CALL_choice(m2_choice_point)

    myham.episodic_reset("initial observation")

    machine_state, reward, done, info = myham.start(loop_machine)

    m_stack = machine_state.m
    self.assertTrue(np.array_equal(np.stack(m_stack), np.array([0,1])))
    for i in range(1,10):
      machine_state, reward, done, info = myham.step("my choice")
      m_stack = machine_state.m
      self.assertTrue(np.array_equal(np.stack(m_stack), np.array([0,1])))
    myham.terminate()

  def test_reassign_machine(self):
    myham = HAM(self.env_exe, "onehot")
    m2_choice_point = myham.choicepoint("m2_choice_point", spaces.Discrete(2), 1)

    @myham.machine
    def loop_machine(ham: HAM):
      for i in range(10):
        ham.CALL(machine2)

    @myham.machine
    def machine2(ham: HAM):
      ham.CALL_choice(m2_choice_point)

    myham.episodic_reset("initial observation")

    machine_state, reward, done, info = myham.start(loop_machine)

    m_stack = machine_state.m
    self.assertTrue(np.array_equal(np.stack(m_stack), np.array([[1,0],[0,1]])))
    for i in range(1,10):
      machine_state, reward, done, info = myham.step("my choice")
    myham.terminate()

    @myham.machine
    def root(ham: HAM):
      ham.CALL(loop_machine)
      ham.CALL(loop_machine)
    
    machine_state, reward, done, info = myham.start(root)
    m_stack = machine_state.m
    self.assertTrue(np.array_equal(np.stack(m_stack), np.array([[0,0,1],[1,0,0],[0,1,0]])))
    for i in range(1,10):
      machine_state, reward, done, info = myham.step("my choice")
      m_stack = machine_state.m
      self.assertTrue(np.array_equal(np.stack(m_stack), np.array([[0,0,1],[1,0,0],[0,1,0]])))
    myham.terminate()


class TestEdgeCases(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    self.rng = default_rng(42)
    
    def env_exe(action):
      return "observation", 0.5, action=="end_env", "info"
    self.myham = HAM(env_exe)
    return super().setUp()

  @pytest.mark.timeout(3)
  def test_multiple_start1(self):
    choice_point1 = self.myham.choicepoint("choice_point1", spaces.Discrete(2), 1)
    @self.myham.machine_with_repr(representation="machine1")
    def m1_func(ham, arg1, arg2):
        while ham.is_alive:
            x=ham.CALL_choice("choice_point1")
            ham.CALL_action(x)
    
    self.myham.episodic_reset("initial observation")
    self.myham.start(m1_func, ("args1", "args2"))
    self.myham.start(m1_func, ("args1", "args2"))
    for _ in range(10):
      self.myham.step("action1")
    obsv, rew, done, info = self.myham.step("end_env")
    self.assertTrue(done)

  @pytest.mark.timeout(3)
  def test_double_reset(self):
    choice_point1 = self.myham.choicepoint("choice_point1", spaces.Discrete(2), 1)
    @self.myham.machine_with_repr(representation="machine1")
    def m1_func(ham, arg1, arg2):
        while ham.is_alive:
            x=ham.CALL_choice(choice_point1)
            ham.CALL_action(x)
    
    self.myham.episodic_reset("initial observation")
    self.myham.episodic_reset("initial observation")
    self.assertTrue(True)
    self.myham.start(m1_func, ("args","args2"))
    self.myham.episodic_reset("initial observation")
    self.assertTrue(True)
    self.myham.start(m1_func, ("args","args2"))
    for _ in range(10):
      self.myham.step("action1")
    self.myham.episodic_reset("initial observation")
    self.assertTrue(True)

# class TestMultiChoicepoint(unittest.TestCase): # TODO
