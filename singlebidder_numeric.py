import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def f(c, y, n):
    w = y[0]

    val = (n-1)/2 * ((w-c)/(1-c) - (1-c)/(w-c))
    return val


def solve_opt(n):

    t = np.zeros((1000-1))
    y = np.zeros((1000-1))
    w0 = .998
    y0 = [w0]
    t0 = .999
    t1 = .001
    dt = -0.001
    n=2
    r = ode(f)
    r.set_f_params(n)
    r.set_initial_value(y0, t0)
    r.set_integrator("lsoda", nsteps=5000)
    i = 0
    while r.successful() and r.t > t1:
        r.integrate(r.t+dt)
        # print("%g %g" % (r.t, r.y))
        t[i] = r.t
        y[i] = r.y
        i+=1
    # w = np.where(y<0, 0, y)
    # plt.plot(t, w)

    # plt.show()
    bid_func = interp1d(t[::-1], y[::-1])
    print(min(t), max(t))
    return bid_func


bf = solve_opt(2)

p = np.linspace(0.01, 0.99, num=100)
plt.plot(p, bf(p))
plt.show()