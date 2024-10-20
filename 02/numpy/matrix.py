import numpy as np

a = np.asarray([[1, 2], [3, 4]])

print(a)
print(a.T)
print(a.dot(a))
print(np.linalg.inv(a))
print(a.dot(np.linalg.inv(a)))
print(np.linalg.det(a))
print(np.linalg.eig(a))