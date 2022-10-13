from matplotlib import pyplot as plt
import numpy as np

def f(x):
    return -np.sin(x*x)/x + 0.01 * x*x
    
if __name__=='__main__':
    x = np.linspace(-3,3,100)
    with open('output.dat', 'w') as file:
        for i in range(len(x)):
            file.write(f"{x[i]} {f(x[i])}")
    
    # Plot
    y = f(x)
    assert len(x) == len(y), "Shape mismatch: {} != {}".format(len(x), len(y))
    plt.scatter(x, y, label=r'$- \frac{sin(x^2)}{x} + 0.01 * x^2$')
    plt.plot(x, y, color='lightblue')
    plt.xlim((-1,1))
    plt.title(r'Plot of $- \frac{sin(x^2)}{x} + 0.01 * x^2$')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.savefig("output5.png", bbox_inches='tight')