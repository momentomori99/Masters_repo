import random
import numpy as np

N = 200
x1 = [random.randint(0, 10) for _ in range(N)]
y1 = [0 if x < 5 else 1 for x in x1]




np.save('x1.npy', x1)
np.save('y1.npy', y1)

