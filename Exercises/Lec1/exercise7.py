from ctypes import Union
from typing import Any


class Polynomial():
    def __init__(self, degree) -> None:
        self.degree = degree
        self.params = None
    
    def __call__(self, x: float) -> Any:
        return eval(x)
    
    def set_parameters(self, params) -> None:
        if len(params) != self.degree + 1:
            raise ValueError("Parameter length does not match polynomial degree.")
        self.params = params
    
    def get_parameters(self):
        if self.params is None:
            raise ValueError("Parameters not set.")
        return self.params
    
    def eval(self, x: float) -> float:
        r = 0
        for i in range(0, self.degree):
            r += self.params[i] * x ** i
        return r
        
        
if __name__=='__main__':
    p = Polynomial(2)
    p.set_parameters([1, 2, 3])
    print(f"Parameters {p.get_parameters()}")
    values = (1, 2, 3, 4, 5)
    for value in values:
        print(f"Value {value} -> {p.eval(value)}")
        
    
    
    