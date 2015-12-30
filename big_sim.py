__author__ = 'msbcg452'

import numpy as np
import matplotlib.pyplot as mpl
def br(G1, G2):
    eps = 1e-3
    tol1 = 1000
    tol2 = 1000

    pos1, pos2 = np.unravel_index(G1.argmax(), G1.shape)
    # pos1 = 0
    # pos2 = 0
    # print(G1)
    # print(pos1, pos2)
    payoff1 = G1[pos1, pos2]
    payoff2 = G2[pos1, pos2]
    while (np.abs(tol1) > eps) and (np.abs(tol2) > eps):
        # print( np.abs(tol1))
        # print( np.abs(tol2))

        pos1 = np.argmax(np.array(G1[:, pos2]))
        # print(G1[:, pos2])
        # print(pos1)
        n_payoff1 = G1[pos1, pos2]
        # print(n_payoff1)
        tol1 = n_payoff1 - payoff1
        # print(tol1)
        payoff1 = n_payoff1

        pos2 = np.argmax(G2[pos1, :])
        # print(G2[pos1, :])
        # print(pos2)
        n_payoff2 = G2[pos1, pos2]
        tol2 = n_payoff2 - payoff2
        payoff2 = n_payoff2
        # print(tol2)

    return pos1, payoff1, pos2, payoff2

def br2(G1, G2):
    eps = 1e-3
    tol1 = 1000
    tol2 = 1000


    # pos1 = 0
    # pos2 = 0
    # print(G1)
    # print(pos1, pos2)
    payoff1 = G1[pos1]
    payoff2 = G2[pos2]
    while (np.abs(tol1) > eps) and (np.abs(tol2) > eps):
        # print( np.abs(tol1))
        # print( np.abs(tol2))

        pos1 = np.argmax(np.array(G1[:, pos2]))
        # print(G1[:, pos2])
        # print(pos1)
        n_payoff1 = G1[pos1, pos2]
        # print(n_payoff1)
        tol1 = n_payoff1 - payoff1
        # print(tol1)
        payoff1 = n_payoff1

        pos2 = np.argmax(G2[pos1, :])
        # print(G2[pos1, :])
        # print(pos2)
        n_payoff2 = G2[pos1, pos2]
        tol2 = n_payoff2 - payoff2
        payoff2 = n_payoff2
        # print(tol2)

    return pos1, payoff1, pos2, payoff2


c1_res = 101
c2_res = 101
p1_res = 101
p2_res = 101

c1_range = np.linspace(0.8, 1.2, c1_res)
c2_range = np.linspace(0.8, 1.2, c2_res)
p2_range = np.linspace(0, 1.5, p2_res)
w1_range = np.linspace(0, 1.5, p1_res)
p1_range = np.linspace(0, 1.5, p1_res)
w2_range = np.linspace(0, 1.5, p2_res)
# print(w1_range)
p1_payoffs = np.zeros((p1_res, p1_res))
p2_payoffs = np.zeros((p2_res, p2_res))
# print(p1_range)
w1_matrix = np.tile(w1_range, (p1_res, 1)).T
w2_matrix = np.tile(w2_range, (p2_res, 1))

w1_w2 = w1_matrix - w2_matrix
w2_w1 = w2_matrix - w1_matrix
f1_tie_break_matrix = np.where(w1_w2 < 0, 1, 0)
f1_tie_break_matrix = np.where(w1_w2 == 0, 1/2, f1_tie_break_matrix)
win_prob1 = np.mean(f1_tie_break_matrix, axis=1)
# print(win_prob1)

f2_tie_break_matrix = np.where(w2_w1 < 0, 1, 0)
f2_tie_break_matrix = np.where(w2_w1 == 0, 1/2, f2_tie_break_matrix)
win_prob2 = np.mean(f2_tie_break_matrix, axis=0)
# print(win_prob2)
# print(f1_tie_break_matrix)
# f1_tie_break_matrix = w1_matrix < w2_matrix # make matrix of w1 < w2
# f2_tie_break_matrix = f1_tie_break_matrix.T
# print(f2_tie_break_matrix)
p1_matrix = np.tile(p1_range, (p1_res, 1))
p2_matrix = np.tile(p2_range, (p2_res, 1)).T
f1_price_clear_matrix = w1_matrix <= p1_matrix # price on columns
f2_price_clear_matrix = w2_matrix <= p2_matrix # price on rows
# print(f2_price_clear_matrix)
x = p1_range - np.matrix(np.ones((p1_res, 1))*0.8)
# print(x)
opt_list = []
#
for c1 in c1_range:

    opt_list2 = []
    for c2 in c2_range:
        x = p1_matrix - c1 # payoffs (price on columns)
        y = p2_matrix - c2 # payoffs (price on rows)
        # print(y)
        xx = f1_price_clear_matrix * x
        yy = f2_price_clear_matrix * y
        # print(yy)
        xxx = np.average(xx, axis=1) # find average payoff
        yyy = np.average(yy, axis=0)
        # print(yyy)
        f1_payoffs = (f1_tie_break_matrix.T * xxx).T
        f2_payoffs = f2_tie_break_matrix * yyy
        f1_payoffs = win_prob1 * xxx
        f2_payoffs = win_prob2 * yyy
        # print(f1_payoffs)
        # print(f1_payoffs)
        # print(f2_payoffs)
        # pos1, payoff1, pos2, payoff2 = br(f1_payoffs, f2_payoffs)
        pos1 = np.argmax(f1_payoffs)
        payoff1 = f1_payoffs[pos1]
        # print(pos1, payoff1)
        opt_list2.append(w1_range[pos1])
        # print(opt_list2)
    opt_list.append(np.mean(opt_list2))

mpl.plot(c1_range, opt_list)
mpl.show()
#
# c1 = .8
# c2 = .9
# x = p1_matrix - c1 # payoffs (price on columns)
# y = p2_matrix - c2 # payoffs (price on rows)
# print(y)
# xx = f1_price_clear_matrix * x
# yy = f2_price_clear_matrix * y
# print(yy)
# xxx = np.average(xx, axis=1) # find average payoff
# yyy = np.average(yy, axis=0)
# print(yyy)
# f1_payoffs = (f1_tie_break_matrix.T * xxx).T
# f2_payoffs = f2_tie_break_matrix * yyy
# print(f1_payoffs)
# print(f2_payoffs)
# pos1, payoff1, pos2, payoff2 = br(f1_payoffs, f2_payoffs)
# # print(pos1, payoff1)
