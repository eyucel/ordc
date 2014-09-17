__author__ = 'msbcg452'
from supply2 import simulate
from supply2 import fixed_simulate
from multiprocessing import Pool
import numpy as np
from scipy import optimize
from scipy import stats

def func(bf):
    p = Pool(4)

    simuls = 50000
    jobs = list()
    for i in range(4):
        jobs.append(p.apply_async(simulate, (simuls, bf)))

    val =  -np.mean([j.get() for j in jobs])

    p.close()
    p.join()
    return val

def fixed_func(bf,opp_bf,draws):

    val = -fixed_simulate(draws, own_bf=bf, opp_bf=opp_bf)
    return val


if __name__ == "__main__":
    # p = Pool(4,maxtasksperchild=1000)
    # np.random.seed(5)
    min_bid = .5
    max_bid = 1.5
    delta = max_bid-min_bid
    trials = 1000
    z_hist = np.zeros((trials))
    bidrv = stats.uniform(min_bid, delta)
    for t in range(trials):
        d = bidrv.rvs((1000, 100))
        # print(simulate(1000,.5))
        last_z = 0
        z = 1
        tol = 1e-3
        bracket = .2
        bracket_reduc = .75

        while abs(last_z - z) > tol:
            last_z = z
            lb = (1-bracket) * last_z
            rb = (1+bracket) * last_z
            z = optimize.fminbound(fixed_func, lb, rb, args=(last_z, d), disp=1)
            bracket *= bracket_reduc
        # z = optimize.basinhopping(func,1,disp=True)
        z_hist[t] = z
        print("Trial:", t)

print(z_hist)
print(np.mean(z_hist))