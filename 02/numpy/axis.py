import numpy as np

a = np.asarray([[1, 2, 3], [4, 5, 6]])

print(a)
print(a.shape)
print("axis = 0:", a.sum(axis = 0).shape, "\n", a.sum(axis = 0))
print("axis = 1:", a.sum(axis = 1).shape, "\n", a.sum(axis = 1))

b = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(b)
print(b.shape)
print("axis = 0:", b.sum(axis = 0).shape, "\n", b.sum(axis = 0))
print("axis = 1:", b.sum(axis = 1).shape, "\n", b.sum(axis = 1))
print("axis = 2:", b.sum(axis = 2).shape, "\n", b.sum(axis = 2))

c = np.asarray(range(24)).reshape((4, 3, 2, 1))

print(c)
print(c.shape)
print("axis = 0:", c.sum(axis = 0).shape, "\n", c.sum(axis = 0))
print("axis = 1:", c.sum(axis = 1).shape, "\n", c.sum(axis = 1))
print("axis = 2:", c.sum(axis = 2).shape, "\n", c.sum(axis = 2))
print("axis = 3:", c.sum(axis = 3).shape, "\n", c.sum(axis = 3))