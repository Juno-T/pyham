import unittest
import pyvirtualdisplay

def make_suite():
  testloader = unittest.TestLoader()
  testsuite = testloader.discover("./test", pattern='test_*.py')
  return testsuite

if __name__ == "__main__":
  suite = unittest.TestSuite()
  suite.addTest(make_suite())
  runner = unittest.TextTestRunner(verbosity=2)
  with pyvirtualdisplay.Display(visible=False, size=(1400, 900)) as disp:
    runner.run(suite)
