import numpy as np

a = np.asarray([[1, 2, 3, 4], [5, 6, 7 ,8], [9, 10, 11, 12]])

print(np.mean(a))
print(a.mean())
print(np.mean(a, axis = 0))
print(np.mean(a, axis = 1))

print(np.sum(a))
print(np.sum(a, axis = 0))
print(np.sum(a, axis = 1))
print(np.max(a))
print(np.min(a))