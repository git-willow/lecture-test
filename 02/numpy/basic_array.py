import numpy as np

a = np.asarray([1, 2, 3])
print("1D array", a.shape)
print(a)
b = np.asarray([[1, 2, 3], [4, 5, 6]])
print("2D array", b.shape)
print(b)
c = np.zeros((2, 3), dtype = np.int32)
print("Zero array", c.shape)
print(c)
