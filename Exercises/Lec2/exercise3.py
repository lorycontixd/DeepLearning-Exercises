import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return np.exp(-x) * np.cos(2*np.pi*x)
    
if __name__=='__main__':
    x = np.linspace(0,1,100)
    y = f(x)
    assert len(x) == len(y), "Shape mismatch: {} != {}".format(len(x), len(y))
    fig, ax = plt.subplots(figsize=(12,6))
    plt.title(r'$exp(-x) * cos(2*pi*x)$')
    plt.scatter(x,y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()