import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class OctreeNode:
    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.children = [None] * 8
        self.data = None
        self.is_leaf = False
        self.mass = 0
        self.com = None
def bound_particle(data, center, size):
    bound_points = []
    for point in data:
        x, y, z, _, _ = point
        crit = center[0] - size/2 <= x < center[0] + size/2 and center[1] - size/2 <= y < center[1] + size/2 and center[2] - size/2 <= z < center[2] + size/2
        if crit:
            bound_points.append(point)
    return np.array(bound_points)
def calculate_com(data):
    mass = np.sum(data[:, -2])
    com = np.sum(data[:, :3] * data[:, -2].reshape(-1, 1), axis=0) / mass
    return com
def generate_octree(data, center, size, threshold):
    node = OctreeNode(center, size)
    half_size = size / 2
    child_size = size / 2
    combinations = list(product([-1, 1], repeat=3))
    for i, c in enumerate(combinations):
        x, y, z = c
        child_center = [
        center[0] + x * half_size/2,
        center[1] + y * half_size/2,
        center[2] + z * half_size/2
        ]
        child_data = bound_particle(data, child_center, child_size)
        if child_data.shape[0] != 0:
            if child_data.shape[0] == 1:
                node.children[i] = OctreeNode(child_center, child_size)
                node.children[i].data = child_data
                node.children[i].is_leaf = True
                node.children[i].mass = child_data[0, -2]
                node.children[i].com = child_data[0, :3]
            else:
                child_sum = np.sum(child_data[:, -2])
                if child_data.shape[0] > threshold:
                    node.children[i] = generate_octree(child_data, child_center, child_size, threshold)
                else:
                    node.children[i] = OctreeNode(child_center, child_size)
                    node.children[i].data = child_data
                    node.children[i].is_leaf = True
                node.children[i].mass = child_sum
                node.children[i].com = calculate_com(child_data)
    node.data = child_data
    return node
def get_leaf_nodes(node):
    leaf_nodes = []
    if node is not None:
        if node.is_leaf == True:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                n = get_leaf_nodes(child)
                if n is not None:
                    leaf_nodes.extend(n)
        return leaf_nodes
    else:
        return None
def estimate_center(data, eps=1e-3):
    center = np.mean(data[:, :3], axis=0)
    size = np.max(np.abs(data[:, :3] - center))*2 + eps
    return center, size
# def main():
#     # Generate sample data
#     num_samples = 100
#     data = np.random.rand(num_samples, 4)
#     data[:, -1] = np.abs(data[:, -1]) + 1
#     data = np.concatenate([data, np.arange(data.shape[0]).reshape(-1, 1)], axis=-1)
#     # Extract coordinates and mass
#     x = data[:, 0]
#     y = data[:, 1]
#     z = data[:, 2]
#     mass = data[:, 3]
    
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.scatter(x, y, z, c=mass, cmap='jet')
#     # ax.set_xlabel('X')
#     # ax.set_ylabel('Y')
#     # ax.set_zlabel('Z')
#     # plt.show()
#     center, size = estimate_center(data)
#     head = generate_octree(data, center, size=size, threshold=4)
#     leaf_nodes = get_leaf_nodes(head)
#     leaves = []
#     masses = []
#     for l in leaf_nodes:
#         leaves.extend(l.data)
#         masses.append(l.mass)
# main()