__author__ = 'msbcg452'

import numpy as np
import matplotlib.pyplot as mpl

def eval_player(cost_range, p_matrix, w1_matrix, opponent_bid):
    # p_matrix = np.tile(p_range, (w_res, 1))
    c2_res = opponent_bid.size
    # w1_matrix = np.tile(w_range, (c2_res, 1)).T
    w2_matrix = np.tile(opponent_bid, (w_res, 1))
    # print(opponent_bid)
    # print(w1_matrix)
    # print(w2_matrix)
    w1_w2 = w1_matrix - w2_matrix
    w2_w1 = w2_matrix - w1_matrix
    f1_tie_break_matrix = np.where(w1_w2 < 0, 1, 0)
    f1_tie_break_matrix = np.where(w1_w2 == 0, 1/2, f1_tie_break_matrix)
    # print(f1_tie_break_matrix)

    f1_price_clear_matrix = np.tile(w_range, (p_res, 1)).T <= p_matrix # price on columns
    # print(f1_price_clear_matrix)
    win_prob1 = np.mean(f1_tie_break_matrix, axis=1)
    # print(win_prob1)
    w = []
    for cost in cost_range:
        # print(cost)
        x = p_matrix - cost # payoffs (price on columns)
        # print(x)
        xx = f1_price_clear_matrix * x
        # print(xx)
        # print(xx*win_prob1[:,np.newaxis])
        # xxx = np.average(xx, axis=1) # find average payoff
        # print(xxx)
        # f1_payoffs = win_prob1 * xxx
        # print(f1_payoffs)

        # print(xx*win_prob1)
        f1_payoffs = np.average(xx*win_prob1[:,np.newaxis], axis=1) #better correct way
        pos1 = np.argmax(f1_payoffs)
        # print(f1_payoffs)
        # print(cost, pos1)
        # print(f1_payoffs)
        w.append(w_range[pos1])
    return np.array(w)

c1_res = 500
c2_res = 500
p_res = 500
w_res = 500

c1_range = np.linspace(0.0, 1.0, c1_res)
c2_range = np.linspace(0.0, 1.0, c2_res)
p2_range = np.linspace(0, 1.0, p_res)
w_range  = np.linspace(0, 1.0, w_res)
opt_w1 = np.copy(c1_range)
p_range =  np.linspace(0, 1.0, p_res)
w2_range = np.linspace(0, 1.0, w_res)
opt_w2 = np.copy(c2_range)
p_matrix = np.tile(p_range, (w_res, 1))
print(w_range)
print(c1_range)
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
# mpl.plot(c2_range, opt_w2)
mpl.show()
