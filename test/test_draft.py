from outline import HAM
from numpy.random import default_rng
rng = default_rng(42)

#TODO: make and actual test

def env_exe(action):
    return 1,2,3,4
myham = HAM(env_exe)

@myham.functional_machine
def m1_func(ham, obsv):
    x=ham.CALL(m2_choice)
    print("m2's choice:",x)
    s=0
    for i in range(x+1):
        s+=i
    return s

@myham.learnable_choice_machine
def m2_choice(ham, obsv):
    return rng.integers(10)

@myham.action_machine
def m3_action(ham, obsv):
    action = ham.CALL(m2_choice)
    return action


print(myham.CALL("m1_func"))
print(myham.CALL(m3_action))
print(myham.current_observation)