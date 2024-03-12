import scipy.integrate as si
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from kernels import numerical_kernel, barnes_hut, fmm
class Body:
    def __init__(self, mass, init_loc, init_vel) -> None:
        self.mass = mass
        self.init_loc = init_loc
        self.init_vel = init_vel
def eqm(params,t, *args):
    G = args[0]
    masses = args[1]
    theta = args[2]
    threshold = args[3]
    vel = params[params.shape[0]//2:]
    r = params[:params.shape[0]//2]
    # dvdt = numerical_kernel(masses, r, G)
    dvdt = fmm(masses, r, G, theta, threshold)
    # dvdt = barnes_hut(masses, r, G, theta, threshold)
    dvs=np.concatenate([vel, dvdt])
    return dvs

def play(solution):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    solutions = np.split(solution[..., :solution.shape[-1]//2], solution.shape[-1]//6, axis=-1)
    def animate(i):
        ax.cla() # clear the previous image
        for s in solutions:
            ax.plot(s[:i, 0], s[:i, 1], s[:i, 2])
            ax.scatter(s[i, 0], s[i, 1], s[i, 2], c='r')
    anim = animation.FuncAnimation(fig, animate, interval=1, repeat=False)
    plt.show()

def sim(G, bodies, theta, threshold):
    #(G*1/(5.2*3**0.5))**0.5
    init = np.concatenate(([i.init_loc for i in bodies], [i.init_vel for i in bodies]))
    ts = np.linspace(0, 50, 10000)
    args = (G, np.array([i.mass for i in bodies]), theta, threshold)
    solution = si.odeint(eqm, init.flatten(), ts, args, rtol=1e-10, atol=1e-10)
    return solution

if __name__ == "__main__":
    G = 4 * np.pi**2
    theta = 0.7
    threshold = 3
    num_bodies = 3
    cluster1 = []
    cluster2 = []

    for i in range(num_bodies):
        mass = np.random.uniform(0.1, 0.5)
        init_loc = np.random.uniform(-6, -4, size=3)
        init_vel = np.random.uniform(-0.1, 0.1, size=3)
        body = Body(mass, init_loc, init_vel)
        cluster1.append(body)

    for i in range(num_bodies):
        mass = np.random.uniform(0.1, 0.5)
        init_loc = np.random.uniform(4, 6, size=3)
        init_vel = np.random.uniform(-0.1, 0.1, size=3)
        body = Body(mass, init_loc, init_vel)
        cluster2.append(body)

    bodies = cluster1 + cluster2
    solution = sim(G, bodies, theta, threshold)
    play(solution)