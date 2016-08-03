__author__ = 'msbcg452'
import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
# from scipy.special import gamma
# from scipy.special import hyp2f1
from scipy.special import betainc
from scipy.special import beta
from scipy.special import lambertw
import mpmath as mp
import seaborn as sns
# import matplotlib as mpl
# mpl.rcParams['font.family'] = 'Arial'
np.random.seed(5)
# sns.set(font="Arial")

class Foo(object):
    pass



def bidf(c, n, a):
    # num1 = (a*a)* (-1 + n) * lambertw(-np.exp((2 * (-1 + c) + a * (-1 + n) * (-4 + a + 4 * c - 2 * a * c + 2 * a * (-1 + c) * n))/(a*a * (n-1))))
    # num2 = -2*c + a *(a - 2 * c) * (-1 + n)
    # denom = 2 + 2 * a * (-1 + n)
    # w = -(num1 + num2) /denom

    w = ((1-n)*a*a + 2 * a * c * n - 2 * a * c + 2 * c)/(2 + 2 * a * (-1 + n))

    y = np.where(w < 0, 0, w)
    return y
def f(c, y):
    w = y[0]

    f2 = (2*c* (1 - 2 * c + w) * (-1 + w)) / (2*(c**2 -1) *(c - w))
    f5 = 5*c**4 * (1-2*c+w) * (-1+w)/ (2*(c**5-1)*(c-w))
    f1 = (1-2*c+w) * (-1+w)/ (2*(c-1)*(c-w))
    f3 = (3*c**2 * (1 - 2 * c + w) * (-1 + w)) / (2*(c**3 -1) *(c - w))
    # f0 = (w-2*c+1)*(w-1) /( (c-1)*(c-w))
    return f2
def multi_winner_ode(c, y, n, k):
    w = y[0] #input comes in as an array

    c = mp.mpf(c)
    w = mp.mpf(w)
    num = -2**(-n) * (2-c)**(n-k-1) * c**(k-1) * (2*c - w - 2) * (w-2)
    denom = (c-w) * mp.beta(k, n-k) * mp.betainc(k, n-k, 0, c/2, regularized=True)
    return num/denom


def solve_opt(n,k):
    nsteps = 1000-1
    t = np.zeros((nsteps))
    y = np.zeros((nsteps))
    w0 = 1.98
    y0 = [w0]
    t0 = 1.99
    t1 = .001
    dt = (t1-w0)/nsteps
    r = ode(multi_winner_ode)
    r.set_f_params(n, k)
    r.set_initial_value(y0, t0)
    r.set_integrator("dropri853", nsteps=5000)
    i = 0
    while r.successful() and r.t > t1:
        r.integrate(r.t+dt)
        print("%g %g" % (r.t, r.y))

        t[i] = r.t
        y[i] = r.y
        i+=1
    w = np.where(y<0, 0, y)
    # plt.plot(t, w)

    # plt.show()
    bid_func = interp1d(t[::-1], w[::-1])
    print(min(t), max(t))
    return bid_func
    # print(y[::-1])
def simulate(ps, f ,n,k):
    M = 200000


    # own_idx = np.random.random_integers(0,n-1,size=(M,1))
    # generate random bids
    costs = np.random.rand(M, n)
    sorted_owners = np.argsort(costs, axis=1)
    sorted_costs = np.sort(costs, axis=1)
    sorted_costs[sorted_costs <.001] = .001
    sorted_costs[sorted_costs >1.99] = 1.99
    # print(costs)
    # f = None
    if f is None:
        sorted_bids = sorted_costs
    else:
        sorted_bids = f(sorted_costs)
    # p = np.random.rand(k,2)
    # level 1 price
    # ps = .9
    alpha=.2
    # other prices
    # p = np.array([ps, ps-alpha, ps-2*alpha])
    p = np.array(k*[ps]+(n-k)*[-1])

    # p = np.where(p<0,0,p)
    # print(p)
    # set player 1 bids to something fixed.
    # bids[:, 0] = .75
    # print(bids)
    # return bidder #s in order
    # sorted_bid_owners = np.argsort(bids, axis=1)
    # sorted_bids = np.sort(bids, axis=1)
    # print(sorted_bids)
    # print(sorted_bid_owners)
    # true if the xth unit of supply is less than xth unit of demand
    p_clearing = sorted_bids <= p
    # print(p_clearing)

    # count number of units that clear
    num_clearing = np.sum(p_clearing, axis=1)
    # print(num_clearing)

    # get position of your bid (1st, 2nd, 3rd) comes out as 0, 1, 2
    # ind = np.where(sorted_owners == own_idx)
    ind = np.where(sorted_owners == 0)
    # print(ind)

    # whether or not you were in the cheapest bids list, so if you were cheapest, ind was 0
    # and you would clear as long as a price cleared. return which price you received (1, 2, 3 or 4)
    mask = np.where(num_clearing > ind[1], num_clearing, 0)
    # print(mask)

    q = np.arange(0,M,dtype=int)
    # print(sorted_bids[q, num_clearing])
    #overpayment
    op = np.mean((num_clearing>0)*(p[num_clearing-1]-sorted_bids[q,num_clearing-1]))

    # find where you received 2s
    mask2 = mask == 2
    maskany = mask > 0
    # print(np.mean(maskany))


    # retrun  m, s, nc, pp, op, ab
    # return np.mean(maskany), np.mean(maskany * (p[num_clearing-1]-costs[:,0]), axis=0), np.mean(num_clearing), np.mean((num_clearing>0) * p[num_clearing-1]),op, np.mean((num_clearing >0)*sorted_bids[:, 0])

    data = Foo()
    data.win_prob = np.mean(maskany)
    data.avg_profit = np.mean(maskany * (p[num_clearing-1]-costs[:, 0]), axis=0)
    data.num_clearing = np.mean(num_clearing)
    data.avg_clearing_price = np.mean((num_clearing > 0) * p[num_clearing-1])
    data.avg_lowest_bid = np.mean((num_clearing > 0)*sorted_bids[:, 0])
    # retrun  m, s, nc, pp, op, ab
    return data

    # return np.mean(maskany), np.mean(sorted_bids, axis=0), np.mean(num_clearing)

    # print(sum(mask)/M)r

    # probability that you won 2s
    # print(np.mean(mask2))

def simulate_compare(ps, own_bf, f ,n,k):
    M = 100000


    # own_idx = np.random.random_integers(0,n-1,size=(M,1))
    # generate random bids
    costs = 2*np.random.rand(M, n)
    lex = np.random.rand(M,n)

    bids = np.zeros((M,n))
    lb = 0.01
    ub = 1.98
    costs[costs < lb] = lb
    costs[costs > ub] = ub
    bids[:,0] = own_bf(costs[:, 0])

    bids[:,1:] = f(costs[:, 1:])
    # print(lex)
    sorted_owners = np.lexsort((lex, bids),axis=1)
    # sorted_owners = np.argsort(bids, axis=1)
    # print(bids)
    # print(sorted_owners)
    # sorted_costs = np.sort(costs, axis=1)
    sorted_bids = np.sort(bids, axis=1)
    # print(sorted_bids)
    # raw_input()

    p = np.array(k*[ps]+(n-k)*[-1])

    # p = np.where(p<0,0,p)
    # print(p)
    # set player 1 bids to something fixed.
    # bids[:, 0] = .75
    # print(bids)
    # return bidder #s in order
    # sorted_bid_owners = np.argsort(bids, axis=1)
    # sorted_bids = np.sort(bids, axis=1)
    # print(sorted_bids)
    # print(sorted_bid_owners)
    # true if the xth unit of supply is less than xth unit of demand
    p_clearing = sorted_bids <= p
    # print(p_clearing)

    # count number of units that clear
    num_clearing = np.sum(p_clearing, axis=1)
    # print(num_clearing)

    # get position of your bid (1st, 2nd, 3rd) comes out as 0, 1, 2
    # ind = np.where(sorted_owners == own_idx)
    ind = np.where(sorted_owners == 0)
    # print(ind)

    # whether or not you were in the cheapest bids list, so if you were cheapest, ind was 0
    # and you would clear as long as a price cleared. return which price you received (1, 2, 3 or 4)
    mask = np.where(num_clearing > ind[1], num_clearing, 0)
    # print(mask)

    q = np.arange(0,M,dtype=int)
    # print(sorted_bids[q, num_clearing])
    #overpayment
    op = np.mean((num_clearing>0)*(p[num_clearing-1]-sorted_bids[q,num_clearing-1]))

    # find where you received 2s
    mask2 = mask == 2
    # find where you received any price
    maskany = mask > 0
    # print(np.mean(maskany))


    # retrun  m, s, nc, pp, op, ab
    # return np.mean(maskany), np.mean(maskany * (p[num_clearing-1]-costs[:,0]), axis=0), np.mean(num_clearing), np.mean((num_clearing>0) * p[num_clearing-1]),op, np.mean((num_clearing >0)*sorted_bids[:, 0])

    data = Foo()
    data.win_prob = np.mean(maskany)
    data.avg_profit = np.mean(maskany * (p[num_clearing-1]-costs[:, 0]), axis=0)
    data.num_clearing = np.mean(num_clearing)
    data.avg_clearing_price = np.mean((num_clearing > 0) * p[num_clearing-1])
    data.avg_lowest_bid = np.mean((num_clearing > 0)*sorted_bids[:, 0])
    # retrun  m, s, nc, pp, op, ab
    return data

    # return np.mean(maskany), np.mean(sorted_bids, axis=0), np.mean(num_clearing)

    # print(sum(mask)/M)r

    # probability that you won 2s
    # print(np.mean(mask2))
def run():
    bf = None


    n = 6
    k = 4
    z = np.genfromtxt("n6k4.csv",delimiter=',')
    bf = interp1d(z[:,0],z[:,1])

    ps = np.linspace(.01, 1.98)
    # plt.plot(ps, bf(ps))
    d1_list = []
    d2_list = []
    np.random.seed(10)
    for p in ps:
        np.random.seed(10)
        d1 = simulate(p, bf,n,k)
        d1_list.append(d1.avg_profit)
        np.random.seed(10)
        d2 = simulate(p, lambda x: x, n, k)
        d2_list.append(d2.avg_profit)

    # print(d1.avg_profit)
    # print(d2.avg_profit)
    plt.plot(ps, d1_list)
    plt.plot(ps, d2_list)
    plt.show()
def run_compare():
    bf = None


    n = 6
    k = 4
    z = np.genfromtxt("n6k4.csv",delimiter=',')
    bf = interp1d(z[:,0],z[:,1])

    ps = np.linspace(.01, 1.98)
    # plt.plot(ps, bf(ps))
    d1_list = []
    d2_list = []
    np.random.seed(10)
    i = 1
    for p in ps:
        np.random.seed(i)
        d1 = simulate_compare(p, bf, lambda x: x ,n,k)
        d1_list.append(d1.avg_profit)
        np.random.seed(i)
        d2 = simulate_compare(p, lambda x: x, lambda x: x, n, k)
        d2_list.append(d2.avg_profit)
        i+=1
    # print(d1.avg_profit)
    # print(d2.avg_profit)
    plt.plot(ps, d1_list)
    plt.plot(ps, d2_list)
    print(np.mean(d1_list)-np.mean(d2_list))
    plt.show()
if __name__ == "__main__":
    # run()
    run_compare()