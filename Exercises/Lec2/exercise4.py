from matplotlib import pyplot as plt
import numpy as np


if __name__=='__main__':
    x,y = np.loadtxt("data4.dat", delimiter=" ", unpack=True)

    fig, ax = plt.subplots(figsize=(12,6))
    plt.title("Charged particles")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.scatter(x,y, color="red")
    plt.savefig("output.png", bbox_inches='tight')

