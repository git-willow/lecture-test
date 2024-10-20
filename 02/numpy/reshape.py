import numpy as np

a = np.asarray([[1, 2, 3, 4], [5, 6, 7 ,8], [9, 10, 11, 12]])

print(a.shape)
print(a)

print(np.reshape(a, (12)))
print(np.reshape(a, (2, 6)))
print(np.reshape(a, (6, 2)))
print(np.reshape(a, (2, -1)))
print(a.reshape(12))