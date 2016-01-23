__author__ = 'msbcg452'

import numpy as np
import matplotlib.pyplot as mpl
import scipy.stats
import scipy.integrate
import scipy.optimize

# def payoff(w, c, p):
#     return (p-c)*(w < p)*np.sum(w<opponent_bid)*price_pdf(p)



def eval_player(cost_range, p_matrix, w1_matrix, opponent_bid):
    def payoff(w, c, p):
        return (p-c)*(w < p)*cost_pdf(c)*price_pdf(p)
        # return (p-c)*(w < p)*np.sum(w < opponent_bid)/np.size(opponent_bid)*price_pdf(p)
    opt_list = []
    for c in cost_range:
        G = lambda w: -scipy.integrate.quad(lambda p: payoff(w, c, p), 0, 1.5)[0]
        # print(G(0))
        res = scipy.optimize.minimize_scalar(G)
        opt_list.append(res.x)
        # print(c)
        # print(res.x)
    return np.array(opt_list)


c1_res = 100
c2_res = 100
p_res = 500
w_res = 500

a = 0
b = 1
s = 0
t = 1

c1_range = np.linspace(s, t, c1_res)
c2_range = np.linspace(s, t, c2_res)
p2_range = np.linspace(a, b, p_res)
w_range  = np.linspace(a, b, w_res)
opt_w1 = np.copy(c1_range)
p_range =  np.linspace(a, b, p_res)
w2_range = np.linspace(a, b, w_res)
opt_w2 = np.copy(c2_range)
p_matrix = np.tile(p_range, (w_res, 1))
print(w_range)
print(c1_range)
price_pdf = lambda x: scipy.stats.uniform.pdf(x, loc=a, scale=b-a)
cost_pdf = lambda x: scipy.stats.uniform.pdf(x, loc=s, scale=t-s)
# f1_payoffs = np.zeros((p_res, p_res))
# f2_payoffs = np.zeros((p_res, p_res))
# # print(p1_range)
# w1_matrix = np.tile(opt_w1, (p_res, 1)).T
# w2_matrix = np.tile(opt_w2, (p_res, 1)).T
#
# w1_w2 = w1_matrix - w2_matrix
# w2_w1 = w2_matrix - w1_matrix
# f1_tie_break_matrix = np.where(w1_w2 < 0, 1, 0)
# f1_tie_break_matrix = np.where(w1_w2 == 0, 1/2, f1_tie_break_matrix)
# win_prob1 = np.mean(f1_tie_break_matrix, axis=1)
# # print(win_prob1)
#
# f2_tie_break_matrix = np.where(w2_w1 < 0, 1, 0)
# f2_tie_break_matrix = np.where(w2_w1 == 0, 1/2, f2_tie_break_matrix)
# win_prob2 = np.mean(f2_tie_break_matrix, axis=1)
# # print(win_prob2)
# # print(f1_tie_break_matrix)
# # f1_tie_break_matrix = w1_matrix < w2_matrix # make matrix of w1 < w2
# # f2_tie_break_matrix = f1_tie_break_matrix.T
# # print(f2_tie_break_matrix)
# p_matrix = np.tile(p_range, (p_res, 1))
# f1_price_clear_matrix = w1_matrix <= p_matrix # price on columns
# f2_price_clear_matrix = w2_matrix <= p_matrix # price on rows
# # print(f2_price_clear_matrix)


tol1 = 1e3
tol2 = 1e3
eps = 1e-5
tol2 = 0
w_matrix = np.tile(w_range, (c2_res, 1)).T
while (tol1+tol2) > eps:
    new_opt_w1 = eval_player(c1_range, p_matrix, w_matrix, opt_w1)
    tol1 = np.sum(new_opt_w1 - opt_w1)**2
    # print(new_opt_w1 - opt_w1)
    print(tol1)
    opt_w1 = new_opt_w1
    # print(opt_w1)

    # new_opt_w2 = eval_player(c2_range, p_range, w2_range, opt_w1)
    # tol2 = np.sum(new_opt_w2 - opt_w2)**2
    # opt_w2 = new_opt_w2
    # # print(opt_w2)
    # # f1 = mpl.figure()
mpl.plot(c1_range, opt_w1)
mpl.axis([s, t, a, b])
# mpl.plot(c2_range, opt_w2)
mpl.show()
print(opt_w1[0],opt_w1[-1])