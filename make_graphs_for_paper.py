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
import time

import mpmath as mp
import seaborn as sns
import matplotlib as mpl
# mpl.rcParams['font.family'] = 'Arial'
np.random.seed()
sns.set_context("paper", font_scale=1.75, rc={"lines.linewidth": 2.5})
sns.set_palette(sns.cubehelix_palette(5, reverse=True))
mpl.rcParams['svg.fonttype'] = 'none'
# sns.set(font="Arial")
def bidf(c, n, a):
    # num1 = (a*a)* (-1 + n) * lambertw(-np.exp((2 * (-1 + c) + a * (-1 + n) * (-4 + a + 4 * c - 2 * a * c + 2 * a * (-1 + c) * n))/(a*a * (n-1))))
    # num2 = -2*c + a *(a - 2 * c) * (-1 + n)
    # denom = 2 + 2 * a * (-1 + n)
    # w = -(num1 + num2) /denom

    w = ((1-n)*a*a + 2 * a * c * n - 2 * a * c + 2 * c)/(2 + 2 * a * (-1 + n))

    y = np.where(w<0, 0, w)
    return y
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
    r.set_integrator("lsoda", nsteps=5000)
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
    print(min(t), max(t))
    return bid_func
    # print(y[::-1])
def simulate(ps, f,n,k):
    M = 200000


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

    # print(p)
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
    M = 200000

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
    q = np.arange(0, M, dtype=int)
    # print(sorted_bids[q, num_clearing])

    #overpayment
    op = np.mean((num_clearing > 0)*(p[num_clearing-1]-sorted_bids[q, num_clearing-1]))


    # retrun  m, s, nc, pp, op, ab
    return (np.mean(maskany),  # individual probability of winning
           np.mean(maskany * (p[num_clearing-1]-costs[:, 0]), axis=0),  # avg profit to bidder
           np.mean(num_clearing),  # average number clearing
           np.mean((num_clearing > 0) * p[num_clearing-1]),  # avg clearing price
           op,
           np.mean((num_clearing > 0)*sorted_bids[:, 0]))  # avg lowest bid)

    # print(sum(mask)/M)r

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
    p = np.linspace(0.01, 0.99, num=100)
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
    p = np.linspace(0.01, 0.99, num=100)
    for ps in p:
        m,s,nc = simulate(ps, bf, n, k)
        m_list.append(m)
        # print(s)
        s_list.append(s[0:k])
        nc_list.append(nc)
    plt.figure()
    plt.plot(p, s_list)
    plt.show()

def report2():

    file_time = time.strftime("%Y%H%M%S")
    bf = None
    n = 3
    # bf = solve_opt(n, k)
    bf = bidf
    a = .3
    for n in range(2,6):
        print(bidf(1,n,a))
        c = np.linspace(.001,.998,num=100)
        # f4 = plt.figure(4)
        # plt.plot(c,bf(c,n,a=.2),label='n='+str(n))
        # plt.title('Optimal Bid Function')
        # plt.xlabel('Cost')
        # plt.ylabel('Bid')
        # ax = f4.gca()
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, loc=0)
        # f4.savefig('n_bf_desc.pdf')
        m_list = []
        s_list = []
        nc_list = []
        cp_list = []
        op_list = []
        ab_list=[]
        p = np.linspace(0.01, 0.99, num=100)
        for ps in p:
            m, s, nc, pp, op, ab = simulate2(ps, bf, n, a)
            # print(pp, ab)
            # input()
            m_list.append(m)
            # print(s)
            s_list.append(s)
            nc_list.append(nc)
            cp_list.append(pp)
            op_list.append(op)
            ab_list.append(ab)
        #########################
        ## individual win prob ##
        #########################
        f7 = plt.figure(7)
        plt.plot(p, m_list, label='n='+str(n))
        # plt.title('Probability of Individual Winning')
        plt.xlabel('Price')
        plt.ylabel('Probability')

        ####################
        ## average profit ##
        ####################
        # f1 = plt.figure(1)
        # plt.plot(p, s_list, label='n='+str(n))
        # # plt.title('Avg. Profit')
        # plt.xlabel('Price')
        # plt.ylabel('Profit')
        # # plt.figure(1)
        # # plt.plot(p, s_list, label='n='+str(n))
        # # plt.title('accepted bids')


        #########################
        ## mean units accepted ##
        #########################
        f2 = plt.figure(2)
        plt.plot(p, nc_list, label='n='+str(n))
        # plt.title('Mean Number of Units Accepted')
        plt.xlabel('Price')
        plt.ylabel('Units')
        # plt.figure()
        # plt.plot(p,np.gradient(nc_list,np.gradient(p)))
        # plt.title("derivative")
        # plt.axis([0,1,0,1.1])

        ############################
        ## cost to auction holder ##
        ############################

        # f3 = plt.figure(3)
        # # cost to auction holder
        # plt.plot(p, np.array(p_list)*np.array(nc_list), label='n='+str(n))
        # # plt.title('Total Expected Cost')
        # plt.xlabel('Price')
        # plt.ylabel('Cost')


        ########################
        ## avg clearing price ##
        ########################

        f5 = plt.figure(5)
        plt.plot(p, cp_list, label='n='+str(n))
        # plt.title('Avg Clearing Price')
        plt.xlabel('Initial Price')
        plt.ylabel('Final Clearing Price')
        # plt.show()

        #################
        ## overpayment ##
        #################
        f6 = plt.figure(6)
        plt.plot(p, np.array(op_list)*np.array(nc_list), label='n='+str(n))
        # plt.title(Overpayment')
        plt.xlabel('Initial Price')
        plt.ylabel('Excess Payment')
        # # plt.show()

        ##########################
        ## optimal bid function ##
        ##########################
        c = np.linspace(.001,.998,num=100)
        f4 = plt.figure(4)
        plt.plot(c, bf(c,n,a), label='n = '+str(n))
        # plt.title('Optimal Bid Function')
        plt.xlabel('Cost')
        plt.ylabel('Bid')
        # plt.show()

    ax = f2.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 4])
    f2.savefig("desc_units_"+file_time+".pdf",bbox_inches='tight')

    ax = f4.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f4.savefig("desc_bf_"+file_time+".pdf",bbox_inches='tight')

    ax = f5.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f5.savefig("desc_clearing_price_"+file_time+".pdf",bbox_inches='tight')

    ax = f6.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f6.savefig("desc_overpayment_"+file_time+".pdf",bbox_inches='tight')

    ax = f7.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f7.savefig("desc_win_prob_"+file_time+".pdf",bbox_inches='tight')
    plt.show()

def report():
    bf = None
    n = 3
    k = 4
    # bf = solve_opt(n, k)
    bf = bidf
    m_list = []
    s_list = []
    nc_list = []
    p = np.linspace(0.01, 0.99, num=100)
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
    # plt.figure()
    # plt.plot(p,np.gradient(nc_list,np.gradient(p)))
    # plt.title("derivative")
    # plt.axis([0,1,0,1.1])
    plt.figure()
    plt.plot(p, p*np.array(nc_list))
    plt.title('total payments')
    plt.show()

def reportsingle():
    bf = None
    for n in range(2,6):
        k = 1
        bf = lambda c: np.where((2*n*c - (n-1))/(n+1)<0,0,(2*n*c - (n-1))/(n+1))
        m_list = []
        s_list = []
        nc_list = []
        p_list = []
        op_list = []
        ab_list=[]
        p = np.linspace(0.01, 0.99, num=100)
        for ps in p:
            m,s,nc, pp, op,ab  = simulate(ps, bf, n, k)
            m_list.append(m)
            # print(s)
            s_list.append(s)
            nc_list.append(nc)
            p_list.append(pp)
            op_list.append(op)
            ab_list.append(ab)


        #########################
        ## individual win prob ##
        #########################
        f7 = plt.figure(7)
        plt.plot(p, m_list, label='n='+str(n))
        # plt.title('Probability of Individual Winning')
        plt.xlabel('Price')
        plt.ylabel('Probability')

        ####################
        ## average profit ##
        ####################
        # f1 = plt.figure(1)
        # plt.plot(p, s_list, label='n='+str(n))
        # # plt.title('Avg. Profit')
        # plt.xlabel('Price')
        # plt.ylabel('Profit')
        # # plt.figure(1)
        # # plt.plot(p, s_list, label='n='+str(n))
        # # plt.title('accepted bids')


        #########################
        ## mean units accepted ##
        #########################
        f2 = plt.figure(2)
        plt.plot(p, nc_list, label='n='+str(n))
        # plt.title('Mean Number of Units Accepted')
        plt.xlabel('Price')
        plt.ylabel('Units')
        # plt.figure()
        # plt.plot(p,np.gradient(nc_list,np.gradient(p)))
        # plt.title("derivative")
        # plt.axis([0,1,0,1.1])

        ############################
        ## cost to auction holder ##
        ############################

        # f3 = plt.figure(3)
        # # cost to auction holder
        # plt.plot(p, np.array(p_list)*np.array(nc_list), label='n='+str(n))
        # # plt.title('Total Expected Cost')
        # plt.xlabel('Price')
        # plt.ylabel('Cost')


        ########################
        ## avg clearing price ##
        ########################

        # f5 = plt.figure(5)
        # plt.plot(p, ab_list, label='n='+str(n))
        # # plt.title('Avg Clearing Price')
        # plt.xlabel('Initial Price')
        # plt.ylabel('Final Clearing Price')
        # # plt.show()

        #################
        ## overpayment ##
        #################
        # f6 = plt.figure(6)
        # plt.plot(p, np.array(op_list)*np.array(nc_list), label='n='+str(n))
        # # plt.title(Overpayment')
        # plt.xlabel('Initial Price')
        # plt.ylabel('Excess Payment')
        # # plt.show()

        ##########################
        ## optimal bid function ##
        ##########################
        c = np.linspace(.001,.998,num=100)
        f4 = plt.figure(4)
        plt.plot(c,bf(c),label='n = '+str(n))
        # plt.title('Optimal Bid Function')
        plt.xlabel('Cost')
        plt.ylabel('Bid')
        # plt.show()

    ax = f2.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 4])
    f2.savefig('single_units.pdf',bbox_inches='tight')

    ax = f4.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f4.savefig("single_bf.pdf",bbox_inches='tight')

    ax = f7.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f7.savefig("single_win_prob.pdf",bbox_inches='tight')
    plt.show()
def reportn():
    bf = None
    for n in range(3,6):
        k = 2
        bf = solve_opt(n, k)
        m_list = []
        s_list = []
        nc_list = []
        p_list = []
        op_list = []
        ab_list=[]
        p = np.linspace(0.01, 0.99, num=100)
        for ps in p:
            m,s,nc, pp, op,ab  = simulate(ps, bf, n, k)
            m_list.append(m)
            # print(s)
            s_list.append(s)
            nc_list.append(nc)
            p_list.append(pp)
            op_list.append(op)
            ab_list.append(ab)
        f7 = plt.figure(7)
        plt.plot(p, m_list, label='n='+str(n))
        # plt.title('Probability of Individual Winning')
        plt.xlabel('Price')
        plt.ylabel('Probability')

        f1 = plt.figure(1)
        plt.plot(p, s_list, label='n='+str(n))
        # plt.title('Avg. Profit')
        plt.xlabel('Price')
        plt.ylabel('Profit')
        # plt.figure(1)
        # plt.plot(p, s_list, label='n='+str(n))
        # plt.title('accepted bids')
        f2 = plt.figure(2)
        plt.plot(p, nc_list, label='n='+str(n))
        # plt.title('Mean Number of Units Accepted')
        plt.xlabel('Price')
        plt.ylabel('Units')
        # plt.figure()
        # plt.plot(p,np.gradient(nc_list,np.gradient(p)))
        # plt.title("derivative")
        # plt.axis([0,1,0,1.1])
        f3 = plt.figure(3)
        # cost to auction holder
        plt.plot(p, np.array(p_list)*np.array(nc_list), label='n='+str(n))
        # plt.title('Total Expected Cost')
        plt.xlabel('Price')
        plt.ylabel('Cost')
        f5 = plt.figure(5)
        plt.plot(p, ab_list, label='n='+str(n))
        # plt.title('Avg Clearing Price')
        plt.xlabel('Initial Price')
        plt.ylabel('Final Clearing Price')
        # plt.show()
        f6 = plt.figure(6)
        plt.plot(p, np.array(op_list)*np.array(nc_list), label='n='+str(n))
        # plt.title(Overpayment')
        plt.xlabel('Initial Price')
        plt.ylabel('Excess Payment')
        # plt.show()
        c = np.linspace(.001,.998,num=100)
        f4 = plt.figure(4)
        plt.plot(c,bf(c),label='n = '+str(n))
        # plt.title('Optimal Bid Function')
        plt.xlabel('Cost')
        plt.ylabel('Bid')
        # plt.show()
    ax = f1.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    # f1.savefig('n_prob.pdf')
    ax = f2.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    # f2.savefig('n_desc_units.pdf')
    ax = f3.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax = f4.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    f4.savefig("n_bf.pdf",bbox_inches='tight')
    ax = f5.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    # f5.savefig('n_desc_avg_price.pdf')
    ax = f6.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    # f6.savefig('n_desc_tot_op.pdf')
    ax = f7.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    plt.show()
def reportk():
    bf = None

    for k in range(1,5):
        n = 5
        bf = solve_opt(n, k)
        m_list = []
        s_list = []
        nc_list = []
        p = np.linspace(0.01, 0.99, num=100)
        for ps in p:
            m, s, nc = simulate(ps, bf, n, k)
            m_list.append(m)
            # print(s)
            s_list.append(s[0:k])
            nc_list.append(nc)
        f1 = plt.figure(1)
        plt.plot(p, m_list, label='k='+str(k))
        # plt.title('Probability of Individual Winning')
        plt.xlabel('Price')
        plt.ylabel('Probability')
        # plt.figure(1)
        # plt.plot(p, s_list, label='n='+str(n))
        # plt.title('accepted bids')
        f2 = plt.figure(2)
        plt.plot(p, nc_list, label='k='+str(k))
        # plt.title('Mean Number of Units Accepted')
        plt.xlabel('Price')
        plt.ylabel('Units')
        # plt.figure()
        # plt.plot(p,np.gradient(nc_list,np.gradient(p)))
        # plt.title("derivative")
        # plt.axis([0,1,0,1.1])
        f3 = plt.figure(3)
        plt.plot(p, p*np.array(nc_list), label='k='+str(k))
        plt.title('Total Expected Cost')
        plt.xlabel('Price')
        plt.ylabel('Cost')

        c = np.linspace(.001,.998,num=100)
        f4 = plt.figure(4)
        plt.plot(c,bf(c),label='k='+str(k))
        # plt.title('Optimal Bid Function')
        plt.xlabel('Cost')
        plt.ylabel('Bid')
        # plt.show()
    ax = f1.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    f1.savefig('k_prob.pdf')
    ax = f2.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    f2.savefig('k_units.pdf')
    ax = f3.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax = f4.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    f4.savefig('k_bf.pdf')
    plt.close()

def reportk2():
    bf = None
    for k in range(2, 5):
        n = 5
        bf = solve_opt(n, k)
        m_list = []
        s_list = []
        nc_list = []
        p_list = []
        op_list = []
        ab_list=[]
        p = np.linspace(0.01, 0.99, num=100)
        for ps in p:
            m,s,nc, pp, op, ab  = simulate(ps, bf, n, k)
            m_list.append(m)
            # print(s)
            s_list.append(s)
            nc_list.append(nc)
            p_list.append(pp)
            op_list.append(op)
            ab_list.append(ab)

        #########################
        ## individual win prob ##
        #########################
        f7 = plt.figure(7)
        plt.plot(p, m_list, label='k='+str(k))
        # plt.title('Probability of Individual Winning')
        plt.xlabel('Price')
        plt.ylabel('Probability')

        ####################
        ## average profit ##
        ####################
        # f1 = plt.figure(1)
        # plt.plot(p, s_list, label='n='+str(n))
        # # plt.title('Avg. Profit')
        # plt.xlabel('Price')
        # plt.ylabel('Profit')
        # # plt.figure(1)
        # # plt.plot(p, s_list, label='n='+str(n))
        # # plt.title('accepted bids')


        #########################
        ## mean units accepted ##
        #########################
        f2 = plt.figure(2)
        plt.plot(p, nc_list, label='k='+str(k))
        # plt.title('Mean Number of Units Accepted')
        plt.xlabel('Price')
        plt.ylabel('Units')
        # plt.figure()
        # plt.plot(p,np.gradient(nc_list,np.gradient(p)))
        # plt.title("derivative")
        # plt.axis([0,1,0,1.1])

        ############################
        ## cost to auction holder ##
        ############################

        # f3 = plt.figure(3)
        # # cost to auction holder
        # plt.plot(p, np.array(p_list)*np.array(nc_list), label='n='+str(n))
        # # plt.title('Total Expected Cost')
        # plt.xlabel('Price')
        # plt.ylabel('Cost')


        ########################
        ## avg clearing price ##
        ########################

        # f5 = plt.figure(5)
        # plt.plot(p, ab_list, label='n='+str(n))
        # # plt.title('Avg Clearing Price')
        # plt.xlabel('Initial Price')
        # plt.ylabel('Final Clearing Price')
        # # plt.show()

        #################
        ## overpayment ##
        #################
        # f6 = plt.figure(6)
        # plt.plot(p, np.array(op_list)*np.array(nc_list), label='n='+str(n))
        # # plt.title(Overpayment')
        # plt.xlabel('Initial Price')
        # plt.ylabel('Excess Payment')
        # # plt.show()

        ##########################
        ## optimal bid function ##
        ##########################
        c = np.linspace(.001,.998,num=100)
        f4 = plt.figure(4)
        plt.plot(c, bf(c),label='k = '+str(k))
        # plt.title('Optimal Bid Function')
        plt.xlabel('Cost')
        plt.ylabel('Bid')
        # plt.show()

    ax = f2.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 4])
    f2.savefig('k_units.pdf',bbox_inches='tight')

    ax = f4.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f4.savefig("k_bf.pdf",bbox_inches='tight')

    ax = f7.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f7.savefig("k_win_prob.pdf",bbox_inches='tight')
    plt.show()
def reportn2():
    bf = None
    for n in range(3, 6):
        k=2
        bf = solve_opt(n, k)
        m_list = []
        s_list = []
        nc_list = []
        p_list = []
        op_list = []
        ab_list=[]
        p = np.linspace(0.01, 0.99, num=100)
        for ps in p:
            m,s,nc, pp, op, ab  = simulate(ps, bf, n, k)
            m_list.append(m)
            # print(s)
            s_list.append(s)
            nc_list.append(nc)
            p_list.append(pp)
            op_list.append(op)
            ab_list.append(ab)

        #########################
        ## individual win prob ##
        #########################
        f7 = plt.figure(7)
        plt.plot(p, m_list, label='n='+str(n))
        # plt.title('Probability of Individual Winning')
        plt.xlabel('Price')
        plt.ylabel('Probability')

        ####################
        ## average profit ##
        ####################
        # f1 = plt.figure(1)
        # plt.plot(p, s_list, label='n='+str(n))
        # # plt.title('Avg. Profit')
        # plt.xlabel('Price')
        # plt.ylabel('Profit')
        # # plt.figure(1)
        # # plt.plot(p, s_list, label='n='+str(n))
        # # plt.title('accepted bids')


        #########################
        ## mean units accepted ##
        #########################
        f2 = plt.figure(2)
        plt.plot(p, nc_list, label='n='+str(n))
        # plt.title('Mean Number of Units Accepted')
        plt.xlabel('Price')
        plt.ylabel('Units')
        # plt.figure()
        # plt.plot(p,np.gradient(nc_list,np.gradient(p)))
        # plt.title("derivative")
        # plt.axis([0,1,0,1.1])

        ############################
        ## cost to auction holder ##
        ############################

        # f3 = plt.figure(3)
        # # cost to auction holder
        # plt.plot(p, np.array(p_list)*np.array(nc_list), label='n='+str(n))
        # # plt.title('Total Expected Cost')
        # plt.xlabel('Price')
        # plt.ylabel('Cost')


        ########################
        ## avg clearing price ##
        ########################

        # f5 = plt.figure(5)
        # plt.plot(p, ab_list, label='n='+str(n))
        # # plt.title('Avg Clearing Price')
        # plt.xlabel('Initial Price')
        # plt.ylabel('Final Clearing Price')
        # # plt.show()

        #################
        ## overpayment ##
        #################
        # f6 = plt.figure(6)
        # plt.plot(p, np.array(op_list)*np.array(nc_list), label='n='+str(n))
        # # plt.title(Overpayment')
        # plt.xlabel('Initial Price')
        # plt.ylabel('Excess Payment')
        # # plt.show()

        ##########################
        ## optimal bid function ##
        ##########################
        c = np.linspace(.001,.998,num=100)
        f4 = plt.figure(4)
        plt.plot(c, bf(c),label='n = '+str(n))
        # plt.title('Optimal Bid Function')
        plt.xlabel('Cost')
        plt.ylabel('Bid')
        # plt.show()

    ax = f2.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 4])
    f2.savefig('n_units.pdf',bbox_inches='tight')

    ax = f4.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f4.savefig("n_bf.pdf",bbox_inches='tight')

    ax = f7.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=0)
    ax.axis([0, 1, 0, 1])
    f7.savefig("n_win_prob.pdf",bbox_inches='tight')

    plt.show()

if __name__ == "__main__":

    reportsingle()
    reportk2()
    reportn2()
    report2()


    t = np.linspace(0,1)

    plt.plot(t, 1-t)
    bid_line_x = [0,.1,.1,.3,.3,.5,.5,.7,.7,.9,.9]
    bid_line_y = [0,0,.1,.1,.3,.3,.45,.45,.7,.7,10]
    plt.plot(bid_line_x,bid_line_y)
    plt.plot((0,.3),(.7,.7),linestyle='--',color='k',linewidth=1)
    plt.plot((0,.5),(.5,.5),linestyle='--',color='k',linewidth=1)
    plt.plot((0,.7),(.3,.3),linestyle='--',color='k',linewidth=1)

    plt.axis([0,1,0,1])
    plt.gca().set_xticks([.1,.3,.5,.7,.9])
    plt.gca().set_xticklabels(['0','1','2','3','4'])
    plt.gca().set_yticks([.3,.5,.7])
    plt.gca().set_yticklabels(['p-2a','p-a','p'])
    plt.ylabel('Capacity Price')
    plt.xlabel('Capacity Addition')
    plt.savefig("curve_intersect.pdf",bbox_inches='tight')

    plt.show()

