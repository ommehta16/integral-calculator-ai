from math import sqrt
from math import sin, cos, tan
from math import asin as arcsin
from math import acos as arccos
from math import atan as arctan
import math

def process_fn(func:str) -> str:
    func = func.replace("^","**").replace(" ","")
    func = func.lower()
    i=0
    while i < len(func)-1:
        if func[i+1] in "qwertyuiopasdfghjklzxcvbnm(" and func[i] in "1234567890)":
            func = func[:i+1] + "*" + func[i+1:]
            i -= 1
        i += 1

    return func

def evaluate(processed_func, x):
    try:
        res = eval(processed_func)
        return res
    except Exception as e:
        raise e

def integrate(func:str, low:float, high:float) -> float:
    processed = process_fn(func)
    bro = 1
    if low > high:
        bro = -1
        low,high = high, low
    ans = 0.0

    curr = low
    step = 0.01

    rn = 0
    prev = 0
    prev_deriv = 0
    hi_step = (high-low)/1000
    while curr <= high:
        try:
            rn = evaluate(processed, curr)
        except Exception as e:
            raise e

        deriv:float = (rn-prev)/step
        
        goof = abs(100-(deriv-prev_deriv))
        step = max(min(goof,hi_step),0.0001)
        curr += step

        ans += step*rn

        prev = rn
        prev_deriv = deriv
    return ans*bro

def do_integrating(func:str, low:float, high:float) -> str:
    try: return str(integrate(func, low, high))
    except Exception as e: return "CNBD"

if __name__ == "__main__":
    func = input("f(x) = ")
    low = float(input())
    high = float(input())
    print("integral of f(x) from low to high --> ")
    print(do_integrating(func, low, high))