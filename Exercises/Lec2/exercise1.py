import numpy as np

a = np.array([[0.5, -1], [-1, 2]], dtype=np.float32)

# Check dimensions
assert a.shape == (2, 2)
assert a.dtype == np.float32

# Create deep copy
b = a.copy().flatten()
b[::2] = 0

print(b)

