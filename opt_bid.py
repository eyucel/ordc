__author__ = 'Emre'
import numpy as np
from scipy import stats

def sim(d,p, x_bf, y_bf):

    n = d.size

    # p = .4
    x = d[:, 0:1]

    y = d[:, 1:]
    # x = np.array([.5, .3,.2])
    # y = np.array([.5, .4, .1])
    own_bid = x_bf(x)
    opp_bid = y_bf(y)
    # print(own_bid)
    # print(opp_bid)
    bids = np.hstack((opp_bid, own_bid))
    # print(bids)
    # input()
    # print(bids)

    sorted_bid_owners = np.argsort(bids,axis=1)
    # print(sorted_bid_owners)
    # input()
    # print(sorted_bid_owners)
    lowest = sorted_bid_owners[:,n-1] == 0

    p_clearing = own_bid <= p

    # print(p_clearing)
    # print(lowest)
    mask = lowest*p_clearing
    # print(mask)

    payoff = ((p-x)*mask)
    average = np.mean(payoff)

    return average

min_bid = 0
max_bid = 1
delta = max_bid-min_bid
trials = 10000
size = 1
competitors = [2]
for c in competitors:
    print("c=",c)
    # for k in range(7):
        # print("k=",k)
    z_hist = np.zeros((trials))

    z_hist = np.zeros((trials))
    y_hist = np.zeros((trials))
    bidrv = stats.uniform(min_bid, delta)
    for t in range(trials):
        d = bidrv.rvs((1, c))
        # print(d)
        p = np.random.random()

        a = lambda v: np.where(v< (c-1)/(2*c), 0, (2*c*v-(c-1))/(c+1))
        b = lambda v: np.where(v < (c)/(2*(c+1)), 0, (2*(c+1)*v-(c))/(c+2))
        aa = lambda v: np.where(v<1/3, 0, (6*v-2)/4)
        bb = lambda v: np.where(v<1/3, 1, (6*v-2)/4)
        cc = lambda v: v
        dd = lambda v: np.where(v < 1/4, 0, 2*v-.5)
        # print(p)
        z = sim(d, p, a, a)
        y = sim(d, p, dd, a)
        z_hist[t] = z
        y_hist[t] = y

print(np.mean(z_hist))
print(np.mean(y_hist))
