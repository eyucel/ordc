__author__ = 'Emre'
__author__ = 'msbcg452'
import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
# from scipy.special import gamma
# from scipy.special import hyp2f1
# from scipy.special import betainc
from scipy.special import beta
from scipy.special import lambertw
import mpmath as mp
import seaborn as sns
# import matplotlib as mpl
# mpl.rcParams['font.family'] = 'Arial'
np.random.seed()
# sns.set(font="Arial")
def bidf(c, n, a):
    # num1 = (a*a)* (-1 + n) * lambertw(-np.exp((2 * (-1 + c) + a * (-1 + n) * (-4 + a + 4 * c - 2 * a * c + 2 * a * (-1 + c) * n))/(a*a * (n-1))))
    # num2 = -2*c + a *(a - 2 * c) * (-1 + n)
    # denom = 2 + 2 * a * (-1 + n)
    # w = -(num1 + num2) /denom

    w = ((1-n)*a*a + 2 * a * c * n - 2 * a * c + 2 * c)/(2 + 2 * a * (-1 + n))

    y = np.where(w<0, 0, w)
    return y

def flex(c,y,n,k):
    j = n-1
    # n here is other number of players aka N-1
    # k is # number of winners
    w = y[0]
    # print(c,w,n,k)
    # num = (1-c)**j * c**k * (-1+w) * (1-2*c+w) * gamma(1+j) * gamma(1-k+j)**2
    # denom = -2 * (1 - c)**k * c * (c - w) * gamma(k) * gamma(1 - k + j) +  2 * (1 - c)**j * c**(1 + k) * (c - w) * gamma(k) * gamma(1+j) * hyp2f1(1, k - j, 1 + k, c/(-1 + c))
    # print(c**(k-1))
    # print(k,j+1-k)
    # print(betainc(c, k, j+1-k))
    # print(betainc(np.float64(.999),2.0,1.0))
    c = mp.mpf(c)
    w = mp.mpf(w)
    num = -(1-c)**(j-k) * c**(k-1) * (w-1) * (1-2*c + w)
    # print(num)
    denom = 2 * beta(k, j+1-k) * (1 - mp.betainc(k, j+1-k, 0, c, regularized=True)) * (c-w)
    # print(num)
    # print(denom)
    return num/denom


def solve_opt(n,k):

    t = np.zeros((1000-1))
    y = np.zeros((1000-1))
    w0 = .998
    y0 = [w0]
    t0 = .999
    t1 = .001
    dt = -0.001
    r = ode(flex)
    r.set_f_params(n,k)
    r.set_initial_value(y0, t0)
    r.set_integrator("lsoda", nsteps=10000)
    i = 0
    while r.successful() and r.t > t1:
        r.integrate(r.t+dt)
        # print("%g %g" % (r.t, r.y))
        t[i] = r.t
        y[i] = r.y
        i+=1
    w = np.where(y<0, 0, y)
    # plt.plot(t, w)

    # plt.show()
    bid_func = interp1d(t[::-1], w[::-1])
    # print(min(t),max(t))
    return bid_func
    # print(y[::-1])
def simulate(ps, f, n, k):
    M = 50000


    # own_idx = np.random.random_integers(0,n-1,size=(M,1))
    # generate random bids
    costs = np.random.rand(M, n)

    sorted_owners = np.argsort(costs, axis=1)
    sorted_costs = np.sort(costs, axis=1)
    sorted_costs[sorted_costs <.001] = .001
    sorted_costs[sorted_costs >.998] = .998
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
    return np.mean(maskany), np.mean(maskany * (p[num_clearing-1]-costs[:,0]), axis=0), np.mean(num_clearing), np.mean((num_clearing>0) * p[num_clearing-1]),op, np.mean((num_clearing >0)*sorted_bids[:, 0])



    # return np.mean(maskany), np.mean(sorted_bids, axis=0), np.mean(num_clearing)

    # print(sum(mask)/M)r

    # probability that you won 2s
    # print(np.mean(mask2))
def simulate2(ps, f,n,a):
    M = 50000


    # own_idx = np.random.random_integers(0,n-1,size=(M,1))
    # generate random bids
    costs = np.random.rand(M, n)

    sorted_owners = np.argsort(costs, axis=1)
    sorted_costs = np.sort(costs, axis=1)

    # print(costs)
    # f = None
    if f is None:
        sorted_bids = sorted_costs
    else:
        sorted_bids = f(sorted_costs,n,a)
    # p = np.random.rand(k,2)
    # level 1 price
    # ps = .9

    # other prices
    # p = np.array([ps, ps-alpha, ps-2*alpha])
    p = np.array([ps-i*a for i in range(0,n)])
    p = np.where(p<0,0,p)
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

    # find where you received 2s
    mask2 = mask == 2
    maskany = mask > 0
    # print(np.mean(maskany))
    # print(sorted_bids.shape)
    # print(num_clearing.shape)
    q = np.arange(0,M,dtype=int)
    # print(sorted_bids[q, num_clearing])

    #overpayment
    op = np.mean((num_clearing>0)*(p[num_clearing-1]-sorted_bids[q,num_clearing-1]))


    # retrun  m, s, nc, pp, op, ab
    return np.mean(maskany), np.mean(maskany * (p[num_clearing-1]-costs[:,0]), axis=0), np.mean(num_clearing), np.mean((num_clearing>0) * p[num_clearing-1]),op, np.mean((num_clearing >0)*sorted_bids[:, 0])

    # print(sum(mask)/M)r

    # probability that you won 2s
    # print(np.mean(mask2))

class container:
    def __init__(self, name):
        self.name = name


def vollsim(ps, bf_f, bf_s):
    a = .1
    n = 6
    k = 3

    flat = container("flat")
    slope = container("slope")

    container_list = [slope, flat]
    flat.bf = bf_f
    slope.bf = bf_s
    flat.p = np.array(k*[ps]+(n-k)*[-1])
    # print(flat.p)
    slope.p = np.array([ps-i*a for i in range(0, n)])
    slope.p = np.where(slope.p < 0, 0, slope.p)
    M = 50000
    # own_idx = np.random.random_integers(0,n-1,size=(M,1))
    # generate random bids
    costs = np.random.rand(M, n)

    sorted_owners = np.argsort(costs, axis=1)
    sorted_costs = np.sort(costs, axis=1)

    for c in container_list:
        if c.name == "flat":
            sorted_costs[sorted_costs <.001] = .001
            sorted_costs[sorted_costs >.998] = .998
            c.sorted_bids = c.bf(sorted_costs)
        else:
            c.sorted_bids = c.bf(sorted_costs, n, a)
        # p = np.random.rand(k,2)
        # level 1 price
        # ps = .9

        # other prices


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
        c.p_clearing = c.sorted_bids <= c.p
        # print(p_clearing)

        # count number of units that clear
        c.num_clearing = np.sum(c.p_clearing, axis=1)
        # print(num_clearing)

        # get position of your bid (1st, 2nd, 3rd) comes out as 0, 1, 2
        # ind = np.where(sorted_owners == own_idx)
        c.ind = np.where(sorted_owners == 0)
        # print(ind)

        # whether or not you were in the cheapest bids list, so if you were cheapest, ind was 0
        # and you would clear as long as a price cleared. return which price you received (1, 2, 3 or 4)
        c.mask = np.where(c.num_clearing > c.ind[1], c.num_clearing, 0)
        # print(mask)

        # find where you received 2s
        c.mask2 = c.mask == 2
        c.maskany = c.mask > 0
        # print(np.mean(maskany))
        # print(sorted_bids.shape)
        # print(num_clearing.shape)
        c.q = np.arange(0, M, dtype=int)
        # print(sorted_bids[q, num_clearing])

        #overpayment
        c.op = np.mean((c.num_clearing>0)*(c.p[c.num_clearing-1]-c.sorted_bids[c.q, c.num_clearing-1]))
        # retrun  m, s, nc, pp, op, ab

        c.win_prob = np.mean(c.maskany)
        c.profit = np.mean(c.maskany * (c.p[c.num_clearing-1]-costs[:,0]), axis=0)
        c.num_clear = np.mean(c.num_clearing)
        c.clearing_price = np.mean((c.num_clearing>0) * c.p[c.num_clearing-1])
        c.avg_bid = np.mean((c.num_clearing >0)*c.sorted_bids[:, 0])
    return container_list
    # return np.mean(maskany), np.mean(maskany * (p[num_clearing-1]-costs[:,0]), axis=0), np.mean(num_clearing), np.mean((num_clearing>0) * p[num_clearing-1]),op, np.mean((num_clearing >0)*sorted_bids[:, 0])

    # print(sum(mask)/M)r

    # probability that you won 2s
    # print(np.mean(mask2))


if __name__ == "__main__":
    p = np.linspace(0.01, 0.99, num=99, endpoint=True)
    # sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 2.5})

    slope = 0
    flat = 1
    num_clear_list=[[], []]
    lole_list = [None, None]
    n = 6
    k = 3
    bf_f = solve_opt(n, k)
    bf_s = bidf
    for ps in p:
        cl = vollsim(ps, bf_f, bf_s)

        num_clear_list[flat].append(cl[flat].num_clear)

        num_clear_list[slope].append(cl[slope].num_clear)


    lole_calc = lambda irm: 0.1011 * np.power(irm, -49.01)

    lole_list[flat] = lole_calc(0.97 + np.array(num_clear_list[flat])/100)
    lole_list[slope] = lole_calc(0.97 + np.array(num_clear_list[slope])/100)
    f1 = plt.figure(1)
    plt.plot(p, num_clear_list[slope], label='slope')
    plt.plot(p, num_clear_list[flat], label='flat')
    # plt.title('accepted bids')
    plt.xlabel('Price')
    plt.ylabel('Units')
    ax = f1.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)

    f2 = plt.figure(2)
    plt.plot(p, lole_list[slope]*1.45, label='slope')
    plt.plot(p, lole_list[flat]*1.45, label='flat')
    # plt.title('accepted bids')
    plt.xlabel('Price')
    plt.ylabel('Units')
    ax = f1.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)


    plt.show()