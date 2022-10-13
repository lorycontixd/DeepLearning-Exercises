import numpy as np

# Create a numpy array from a python list
a = np.array([[0.5, -1],[-1, 2]], dtype=np.float32)

# Verify array shape
assert a.shape == (2,2)

# Deep copy -> flatten
b = a.copy().flatten()

# Set even elements to 0
b[0::2] = 0

print("a= ",a)
print("b= ",b)