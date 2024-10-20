import numpy as np

a = np.asarray([[1, 2, 3, 4], [5, 6, 7 ,8], [9, 10, 11, 12]])
b = np.ones(4, dtype = np.int32)
print(b)
print(a + a)
print(a * a)

print(a + b)

print(a + 10)
print(a * 10)