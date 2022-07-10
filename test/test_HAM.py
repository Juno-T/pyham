import unittest

from pyham import HAM
from numpy.random import default_rng


class TestFunctionality(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    self.rng = default_rng(42)

    self.replay_buffer = []
    def env_exe(action):
      return "observation", 0.5, False, "info"
    def transition_handler(transition):
      self.replay_buffer.append(transition)
    discount = 1
    self.myham = HAM(env_exe, transition_handler, discount)
    return super().setUp()

  def test_initialize(self):
    @self.myham.learnable_choice_machine
    def m2_choice(ham, obsv):
      return 10

    @self.myham.action_machine
    def m3_action(ham, obsv):
      action = ham.CALL(m2_choice)
      return action

    @self.myham.functional_machine
    def loop_machine(ham, obsv):
      for i in range(10):
        ham.CALL(m3_action)

    self.assertTrue(self.myham.machine_count==3)

  def test_functionality(self):
    @self.myham.learnable_choice_machine
    def m2_choice(ham, obsv):
      return "this is m2 choice"

    @self.myham.action_machine
    def m3_action(ham, obsv):
      action = ham.CALL(m2_choice)
      return action

    @self.myham.functional_machine
    def loop_machine(ham, obsv):
      for i in range(10):
        ham.CALL(m3_action)

    self.assertTrue(self.myham.machine_count==3)
    self.myham.episodic_reset("initial observation")
    self.assertTrue(self.myham._current_observation=="initial observation")
    self.myham.CALL(loop_machine)
    self.assertTrue(len(self.replay_buffer)==9)
    for transition in self.replay_buffer:
      self.assertTrue(transition.a_tm1=="this is m2 choice")
    # Could be more
