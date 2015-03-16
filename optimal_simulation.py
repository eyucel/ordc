__author__ = 'msbcg452'
import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import gamma
from scipy.special import hyp2f1
from scipy.special import betainc
from scipy.special import beta
import mpmath as mp

np.random.seed()

def f(c, y):
    w = y[0]

    f2 = (2*c* (1 - 2 * c + w) * (-1 + w)) / (2*(c**2 -1) *(c - w))
    f5 = 5*c**4 * (1-2*c+w) * (-1+w)/ (2*(c**5-1)*(c-w))
    f1 = (1-2*c+w) * (-1+w)/ (2*(c-1)*(c-w))
    f3 = (3*c**2 * (1 - 2 * c + w) * (-1 + w)) / (2*(c**3 -1) *(c - w))
    # f0 = (w-2*c+1)*(w-1) /( (c-1)*(c-w))
    return f2
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
    r.set_integrator("dop853", nsteps=5000)
    i = 0
    while r.successful() and r.t > t1:
        r.integrate(r.t+dt)
        # print("%g %g" % (r.t, r.y))
        t[i] = r.t
        y[i] = r.y
        i+=1
    w = np.where(y<0, 0, y)
    plt.plot(t, w)

    # plt.show()
    bid_func = interp1d(t[::-1], w[::-1])
    print(min(t),max(t))
    return bid_func
    # print(y[::-1])
def simulate(ps, f,n,k):
    M = 50000


    own_idx = np.random.random_integers(0,n-1,size=(M,1))
    # generate random bids
    costs = np.random.rand(M, n)
    costs[costs <.001] = .001
    costs[costs >.998] = .998
    # print(costs)
    if f is None:
        bids = costs
    else:
        bids = f(costs)
    # p = np.random.rand(k,2)
    # level 1 price
    # ps = .9
    alpha = 0
    # other prices
    # p = np.array([ps, ps-alpha, ps-2*alpha, ps-3*alpha])
    p = np.array(k*[ps]+(n-k)*[-1])

    # print(p)
    # set player 1 bids to something fixed.
    # bids[:, 0] = .75
    # print(bids)
    # return bidder #s in order
    sorted_bid_owners = np.argsort(bids, axis=1)
    sorted_bids = np.sort(bids, axis=1)
    # print(sorted_bids)
    # print(sorted_bid_owners)
    # true if the xth unit of supply is less than xth unit of demand
    p_clearing = sorted_bids <= p
    # print(p_clearing)

    # count number of units that clear
    num_clearing = np.sum(p_clearing, axis=1)
    # print(num_clearing)

    # get position of your bid (1st, 2nd, 3rd) comes out as 0, 1, 2
    ind = np.where(sorted_bid_owners == own_idx)
    # print(ind)

    # whether or not you were in the cheapest bids list, so if you were cheapest, ind was 0
    # and you would clear as long as a price cleared. return which price you received (1, 2, 3 or 4)
    mask = np.where(num_clearing > ind[1], num_clearing, 0)
    # print(mask)

    # find where you received 2s
    mask2 = mask == 2
    maskany = mask > 0
    return np.mean(maskany), np.mean(sorted_bids, axis=0), np.mean(num_clearing)

    # print(sum(mask)/M)r
    # print(np.mean(maskany))
    # probability that you won 2s
    # print(np.mean(mask2))
def multiple_runs():
    bf = None
    n = 5
    for k in range(1,5):
        # k = 1
        plt.subplot(2,2,k)
        bf = solve_opt(n, k)
        m_list = []
        p = np.linspace(.05, .95)
        for ps in p:

            m,s,nc  = simulate(ps, bf, n, k)
            m_list.append(m)

        # print(m_list)
        # num = '22' + str(k)

        plt.plot(p,m_list)
    plt.title('win pct vs price')
    plt.show()

def avg_amount_accepted():
    bf = None
    n = 4
    k = 2
    bf = solve_opt(n, k)
    m_list = []
    s_list = []
    nc_list = []
    p = np.linspace(.05,.95)
    for ps in p:
        m,s,nc = simulate(ps, bf, n, k)
        m_list.append(m)
        # print(s)
        s_list.append(s[0:k])
        nc_list.append(nc)
    plt.figure()
    plt.plot(p,nc_list)
    plt.show()
def avg_winning_bid():
    bf = None
    n = 4
    k = 2
    bf = solve_opt(n, k)
    m_list = []
    s_list = []
    nc_list = []
    p = np.linspace(.05,.95)
    for ps in p:
        m,s,nc = simulate(ps, bf, n, k)
        m_list.append(m)
        # print(s)
        s_list.append(s[0:k])
        nc_list.append(nc)
    plt.figure()
    plt.plot(p, s_list)
    plt.show()

def report():
    bf = None
    n = 6
    k = 5
    bf = solve_opt(n, k)
    m_list = []
    s_list = []
    nc_list = []
    p = np.linspace(.05, .95)
    for ps in p:
        m,s,nc = simulate(ps, bf, n, k)
        m_list.append(m)
        # print(s)
        s_list.append(s[0:k])
        nc_list.append(nc)
    plt.figure()
    plt.plot(p, s_list)
    plt.title('accepted bids')
    plt.figure()
    plt.plot(p, nc_list)
    plt.title('avg addition')
    plt.figure()
    plt.plot(p, p/np.array(nc_list))
    plt.title('average cost $/unit')
    plt.figure()
    plt.plot(p,p*np.array(nc_list))
    plt.title('total payments')
    plt.show()

if __name__ == "__main__":
    report()

    # avg_amount_accepted()
    # avg_winning_bid()
    # multiple_runs()