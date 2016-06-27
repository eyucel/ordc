import numpy as np
import scipy.interpolate
import scipy.stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import starmap
import BSpline
import scipy as sp
class storage:
    def __init__(self):
        pass


def caseA():
    def solve(c_star):
        cs = np.linspace(c_star, uv, nt+1)

        incr = (uv-c_star)/nt
        w_c = np.zeros(nt+1)
        w_c [0] = 0
        for i in range(nt):
            w = w_c[i]
            c = cs[i]
            rhs = 1/2 * ((w-c)/(1-c) - (1-c)/(w-c))

            w_c[i+1] = incr * rhs + w

        bad_form.bids = w_c
        obj = (w_c[-1]-cs[-1])**2
        return obj

    uv = 1
    lv = 0
    nt = 500

    fires = 1

    bad_form = storage()
    step = 200
    c_range = np.linspace(lv, uv, step+1)
    c_star = c_range[1]
    print((uv-lv)/nt)
    incr = (uv-lv)/nt

    objs = np.zeros(step+1)
    objs[0] = 1000
    objs[-1] = 1000
    for i in range(1, step):
        objs[i] = solve(c_range[i])
        print(c_range[i], objs[i])
    z = np.argmin(objs)
    print(z, c_range[z])
    result = minimize(solve, c_range[z], method='nelder-mead', options={'ftol': 1e-8, 'disp': True})
    print(result.x)

def caseB():
    def solve_reverse(w_star):
        n = 2
        b = 1
        cs = np.linspace(lv, res, nt+1)

        incr = (res-c_star)/nt
        w_c = np.zeros(nt+1)
        w_c [0] = 0
        for i in range(nt):
            w = w_c[i]
            c = cs[i]
            rhs = (-(n-1) * f(c)/(1-F(c)) * ((b**2)/2 - (b*c) + c*w - (w**2)/2))/(w-c)
            # print(rhs)
            w_c[i+1] = incr * rhs + w

        bad_form.bids = w_c
        obj = (w_c[-1]-cs[-1])**2
        return obj


    def solve(c_star):
        n = 2
        b = 1
        cs = np.linspace(c_star, res, nt+1)

        incr = (res-c_star)/nt
        w_c = np.zeros(nt+1)
        w_c [0] = 0
        for i in range(nt):
            w = w_c[i]
            c = cs[i]
            rhs = (-(n-1) * f(c)/(1-F(c)) * ((b**2)/2 - (b*c) + c*w - (w**2)/2))/(w-c)
            # print(rhs)
            w_c[i+1] = incr * rhs + w

        bad_form.bids = w_c
        obj = (w_c[-1]-cs[-1])**2
        return obj

    uv = 1
    lv = 0
    nt = 500
    res = 1
    dist = scipy.stats.uniform(loc=.5, scale=1 )

    f = lambda c: dist.pdf(c)
    F = lambda c: dist.cdf(c)
    fires = 1
    uv = res
    bad_form = storage()
    step = 200
    c_range = np.linspace(lv, uv, step+1)
    c_star = c_range[1]
    print((uv-lv)/nt)
    incr = (uv-lv)/nt

    objs = np.zeros(step+1)
    objs[0] = 1000
    objs[-1] = 1000
    for i in range(1, step):
        objs[i] = solve(c_range[i])
        print(c_range[i], objs[i])
    z = np.argmin(objs)
    print(z, c_range[z])
    result = minimize(solve, c_range[z], method='nelder-mead', options={'ftol': 1e-8, 'disp': True})
    print(result.x)

def caseC():
    def solve_reverse(w_star):
        n = 2
        b = 1
        cs = np.linspace(0.5, res, nt+1)

        incr = (res-0.5)/nt
        w_c = np.zeros(nt+1)
        w_c[0] = w_star

        for i in range(nt):
            w = w_c[i]
            c = cs[i]
            rhs = (-(n-1) * f(c)/(1-F(c)) * ((b**2)/2 - (b*c) + c*w - (w**2)/2))/(w-c)

            # print(rhs)
            w_c[i+1] = incr * rhs + w

        bad_form.bids = w_c
        obj = (w_c[-1]-cs[-1])**2
        return obj


    def solve(c_star):
        n = 2
        b = 1
        cs = np.linspace(c_star, res, nt+1)

        incr = (res-c_star)/nt
        w_c = np.zeros(nt+1)
        w_c [0] = 0
        for i in range(nt):
            w = w_c[i]
            c = cs[i]
            rhs = (-(n-1) * f(c)/(1-F(c)) * ((b**2)/2 - (b*c) + c*w - (w**2)/2))/(w-c)
            # print(rhs)
            w_c[i+1] = incr * rhs + w

        bad_form.bids = w_c
        obj = (w_c[-1]-cs[-1])**2
        return obj

    uv = 1
    lv = 0
    nt = 500
    res = 1
    dist = scipy.stats.uniform(loc=.5, scale=1 )

    f = lambda c: dist.pdf(c)
    F = lambda c: dist.cdf(c)
    fires = 1
    uv = res
    bad_form = storage()
    step = 200
    c_range = np.linspace(lv, uv, step+1)
    w_range = np.linspace(0, res, step+1)
    c_star = c_range[1]
    w_star = w_range[0]
    print((uv-lv)/nt)
    incr = (uv-lv)/nt

    objs = np.zeros(step+1)
    objs[0] = 1000
    objs[-1] = 1000
    for i in range(1, step):
        objs[i] = solve_reverse(w_range[i])
        print(w_range[i], objs[i])
    z = np.argmin(objs)
    print(z, w_range[z])
    result = minimize(solve_reverse, w_range[z], method='nelder-mead', options={'ftol': 1e-8, 'disp': True})
    print(result.x)

    n = 2
    b = 1
    cs = np.linspace(0.5, res, nt+1)

    incr = (res-0.5)/nt
    w_c = np.zeros(nt+1)
    w_c [0] = result.x
    for i in range(nt):
        w = w_c[i]
        c = cs[i]
        rhs = (-(n-1) * f(c)/(1-F(c)) * ((b**2)/2 - (b*c) + c*w - (w**2)/2))/(w-c)
        # print(rhs)
        w_c[i+1] = incr * rhs + w

    bad_form.bids = w_c
    plt.plot(cs,w_c)
    plt.show()


if __name__ == "__main__":
    print("hih)")
    caseC()