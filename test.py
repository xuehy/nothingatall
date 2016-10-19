import numpy as np

A = np.array([[1,2],[3,4]])
L = []

for i in range(2):
    a = A[i]
    a[0] += 1
    L.append(a)
print(L)

    
