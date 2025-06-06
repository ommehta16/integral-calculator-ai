# generate random functions to evaluate
from main import *
import random
import os

functions = ["cos", "sin"]
inv = ["arccos", "arcsin", "arctan","tan", "sqrt"]
operations = "*-+/^"

LIMIT = 1e3

def generate(n:int) -> str:
    func = ""
    to_close = 0

    tgt = random.randint(1,n)
    if random.randint(1,5) == 5: func += "x"
    else: func += str((1-2*random.random()) * LIMIT)
    for i in range(tgt):
        func += random.choice(operations)
        c = random.randint(1,100)
        if c < 7:
            func += "("
            if random.randint(1,3) == 1: func += "x"
            else: func += str((1-2*random.random()) * LIMIT)
            to_close+=1
        elif c < 30:
            func += "x"
        elif c < 85:
            func += random.choice(functions) + "("
            if random.randint(1,3) == 3: func += str((1-2*random.random())*LIMIT)+"x"
            else: func += str((1-2*random.random()) * LIMIT)
            to_close+=1
        elif c < 86:
            func += random.choice(inv) + "("
            if random.randint(1,3) == 3: func += str((1-2*random.random())*LIMIT)+"x"
            else: func += str((1-2*random.random()) * LIMIT)
        else:
            func += str((1-2*random.random())*LIMIT)
            if to_close > 0 and random.randint(0,10) != 1:
                func += ")"
                to_close -= 1
    while to_close > 0:
        func += ")"
        to_close-=1
    if "x" not in func:
        func += str((1-2*random.random())*LIMIT)+"x"
    return func

num = 0

while os.path.exists(f"functions{num}.csv"): num+=1
tests = open(f"functions{num}.csv",'w')


for t in range(int(1e4)):
    print(t)
    out = ""
    func = generate(min(t//10 + 5,100))
    for _ in range(100):
        lo = (1-2*random.random())*1e2
        hi = (1-2*random.random())*1e2
        ans = do_integrating(func, lo, hi)
        out += f"{func},{lo},{hi},{ans}\n"
    tests.write(out)