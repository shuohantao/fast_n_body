import matplotlib.pyplot as plt
import numpy as np
from itertools import product
center = [0.5, 0.5, 0.5]
half_size = 0.5
children = []
combinations = list(product([-1, 1], repeat=3))
for i, x, y, z in enumerate(combinations):
    child_center = [
        center[0] + x * half_size/2,
        center[1] + y * half_size/2,
        center[2] + z * half_size/2
    ]
    children.append(child_center)
x = [child[0] for child in children]
y = [child[1] for child in children]
z = [child[2] for child in children]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, cmap='jet')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()