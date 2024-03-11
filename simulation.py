import scipy.integrate as si
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from kernels import numerical_kernel, barnes_hut
class Body:
    def __init__(self, mass, init_loc, init_vel) -> None:
        self.mass = mass
        self.init_loc = init_loc
        self.init_vel = init_vel
def eqm(params,t, *args):
    G = args[0]
    masses = args[1]
    vel = params[params.shape[0]//2:]
    r = params[:params.shape[0]//2]
    dvdt = barnes_hut(masses, r, G, 1, 5)
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
            ax.scatter(s[i-1, 0], s[i-1, 1], s[i-1, 2], c='r')
    anim = animation.FuncAnimation(fig, animate, interval=20)
    plt.show()
def sim(G, bodies):
    #(G*1/(5.2*3**0.5))**0.5
    init = np.concatenate(([i.init_loc for i in bodies], [i.init_vel for i in bodies]))
    ts = np.linspace(0, 100, 3000)
    args = (G, np.array([i.mass for i in bodies]))
    solution = si.odeint(eqm, init.flatten(), ts, args)
    return solution
if __name__ == "__main__":
    G = 4 * np.pi**2
    v = 0.5*(G*1/(5.2*3**0.5))**0.5
    greeks = Body(1, [0, 5.2, 5.2], [-v, 0, 0])
    trojan = Body(0.1, [5.2*3**0.5/2, -2.6, -2.6], [v/2, 0.5*v*3**0.5, v/2])
    jupiter = Body(0.1, [-5.2*3**0.5/2, -2.6, -2.6], [v/2, -0.5*v*3**0.5, v/2])
    saturn = Body(0.2, [0, -9.5, -9.5], [v, 0, 0]) 
    uranus = Body(0.2, [9.5, 0, 0], [0, v, 0]) 
    neptune = Body(0.2, [0, 9.5, 9.5], [-v, 0, 0]) 
    bodies = [Body(np.random.rand(), np.random.rand(3)*10, np.random.rand(3)) for i in range(100)]
    solution = sim(G, bodies)
    play(solution)