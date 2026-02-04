import numpy as np

SIZE = 200

rng = np.random.default_rng()

array_a = rng.random((SIZE, SIZE))
array_b = rng.random((SIZE, SIZE))
array_c = rng.random((SIZE, SIZE))

#print(array_a)
#print(array_b)
print(array_c)

for i in range(SIZE):
    for j in range(SIZE):
        for k in range(SIZE):
            array_c[i,j] += array_a[i,k] * array_b[k,j]


print()
print(array_c)
