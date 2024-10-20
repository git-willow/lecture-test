import numpy as np

a = np.asarray([[1, 2, 3, 4], [5, 6, 7 ,8], [9, 10, 11, 12]])

print(a > 5)
print(a[a > 5])
print(a * (a > 5))

print(np.where(a > 5, a, -1))