from outline import HAM
from numpy.random import default_rng
rng = default_rng(42)

def env_exe(action):
    return 1,2,3,4

@myham.create_functional_machine
def m1_func(ham, obsv):
    x=ham.CALL("m2_choice")
    print("m2's choice:",x)
    s=0
    for i in range(x+1):
        s+=i
    return s

@myham.create_learnable_choice_machine
def m2_choice(ham, obsv):
    return rng.integers(10)


myham = HAM(env_exe)
myham.add_machine("m1_func", m1_func)
myham.add_machine("m2_choice", m2_choice)

print(myham.CALL("m1_func"))