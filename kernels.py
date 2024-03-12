import numpy as np
from octree import *
def numerical_kernel(masses, positions, G):
    positions = positions[np.newaxis, ...]
    positions = np.split(positions, positions.shape[-1]//3, axis=-1)
    positions = np.concatenate(positions, axis=0)
    acc = []
    for i, m in enumerate(masses):
        r = positions - positions[i]
        r = np.delete(r, i, axis=0)
        dvdt = np.sum(G*m*r/np.linalg.norm(r, axis=1).reshape(-1, 1)**3, axis=0)
        acc.append(dvdt)
    return np.concatenate(acc)

def calc_local(masses, positions, G):
    acc = []
    for i, m in enumerate(masses):
        r = positions - positions[i]
        r = np.delete(r, i, axis=0)
        dvdt = np.sum(G*m*r/np.linalg.norm(r, axis=1).reshape(-1, 1)**3, axis=0)
        acc.append(dvdt)
    return np.concatenate(acc)

def numerical_kernel_cell(node, G):
    l = node
    local_masses = l.data[:, -2]
    local_positions = l.data[:, :3]
    local_acc = calc_local(local_masses, local_positions, G)
    return local_acc

def distant_numerical_cell(node, G, r):
    l = node
    distant_masses = l.data[:, -2].reshape(-1, 1)
    distant_positions = l.data[:, :3]
    r1 = distant_positions - r
    dvdt = np.sum(G*distant_masses*r1/np.linalg.norm(r1, axis=1).reshape(-1, 1)**3, axis=0)
    return dvdt

def multipole_force(node, G, r):
    mass = node.mass
    quad = node.quad_moment
    d = node.com - r
    mono_force = G * mass * d / np.linalg.norm(d)**3
    quad_force = np.einsum('ij,j,k->i', quad, d, d) * (3 / np.linalg.norm(d)**5)
    return mono_force + quad_force

def barnes_hut(masses, positions, G, theta, threshold):
    positions = positions[np.newaxis, ...]
    positions = np.split(positions, positions.shape[-1]//3, axis=-1)
    positions = np.concatenate(positions, axis=0)
    masses = masses[..., np.newaxis]
    data = np.concatenate([positions, masses, np.arange(masses.shape[0]).reshape(-1, 1)], axis=-1)
    center, size = estimate_center(data)
    head = generate_octree(data, center, size=size, threshold=threshold)
    leaf_nodes = get_leaf_nodes(head)
    acc = [0]*len(masses)
    for i, l in enumerate(leaf_nodes):
        local_acc = numerical_kernel_cell(l, G)
        for j, d in enumerate(l.data):
            d0 = d[:3]
            id = int(d[-1])
            a = local_acc[j]
            for k, l1 in enumerate(leaf_nodes):
                if k != i:
                    r = l1.com - d0
                    m = l1.mass
                    crit = l1.size / np.linalg.norm(r) < theta
                    if crit:
                        a += G*m*r/np.linalg.norm(r)**3
                    else:
                        a += distant_numerical_cell(l1, G, d0)
            assert acc[id] == 0, "idx error"
            acc[id] = a
    return np.concatenate(acc)

def fmm(masses, positions, G, theta, threshold):
    positions = positions[np.newaxis, ...]
    positions = np.split(positions, positions.shape[-1]//3, axis=-1)
    positions = np.concatenate(positions, axis=0)
    masses = masses[..., np.newaxis]
    data = np.concatenate([positions, masses, np.arange(masses.shape[0]).reshape(-1, 1)], axis=-1)
    center, size = estimate_center(data)
    head = generate_octree(data, center, size=size, threshold=threshold)
    leaf_nodes = get_leaf_nodes(head)
    acc = [0]*len(masses)
    for i, l in enumerate(leaf_nodes):
        local_acc = numerical_kernel_cell(l, G)
        for j, d in enumerate(l.data):
            d0 = d[:3]
            id = int(d[-1])
            a = local_acc[j]
            for k, l1 in enumerate(leaf_nodes):
                if k != i:
                    r = l1.com - d0
                    crit = l1.size / np.linalg.norm(r) < theta
                    if crit:
                        a += multipole_force(l1, G, d0)
                    else:
                        a += distant_numerical_cell(l1, G, d0)
            assert acc[id] == 0, "idx error"
            acc[id] = a
    return np.concatenate(acc)






