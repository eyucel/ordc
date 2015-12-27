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
        payoff1 = n_payoff2
        # print(tol2)

    return pos1, payoff1, pos2, payoff2




c1_res = 41
c2_rest = 41
p1_res = 101
p2_res = 101

c1_range = np.linspace(0, 1, c1_res)
c2_range = np.linspace(0, 1, c2_rest)
p1_range = np.linspace(0, 1, p1_res)
p2_range = np.linspace(0, 1, p2_res)
w1_range = np.linspace(0, 1, p1_res)
w2_range = np.linspace(0, 1, p2_res)
# print(w1_range)
p1_payoffs = np.matrix(np.zeros((p1_res, p1_res)))
p2_payoffs = np.matrix(np.zeros((p2_res, p2_res)))

f1_tie_break_matrix = np.matrix(w1_range+(np.random.random(p1_res)-0.5)/1000).T < np.matrix(w2_range+(np.random.random(p2_res)-0.5)/1000) # make matrix of w1 < w2
f2_tie_break_matrix = f1_tie_break_matrix.T

f1_price_clear_matrix = np.matrix(w1_range).T <= np.matrix(p1_range) # make matrix of w1 < w2
f2_price_clear_matrix = np.matrix(w2_range).T <= np.matrix(p2_range) # make matrix of w1 < w2

x = p1_range - np.matrix(np.ones((p1_res, 1))*0.8)
# print(x)
opt_list = []

for c1 in c1_range:
    opt_list2 = []
    for c2 in c2_range:
        x = p1_range - np.matrix(np.ones((p1_res, 1))*c1) # payoffs (price on columns)
        y = p2_range - np.matrix(np.ones((p2_res, 1))*c2) # payoffs (price on columns)
        # print(x)
        xx = np.multiply(f1_price_clear_matrix, x)
        yy = np.multiply(f2_price_clear_matrix, y)
        # print(xx)
        xxx = np.average(xx, axis=0) # find average payoff
        yyy = np.average(yy, axis=0)
        # print(xxx)
        f1_payoffs = np.multiply(f1_tie_break_matrix, xxx)
        f2_payoffs = np.multiply(f2_tie_break_matrix, yyy).T
        pos1, payoff1, pos2, payoff2  = br(f1_payoffs, f2_payoffs)
        print(pos1, payoff1)
        opt_list2.append(w1_range[pos1])
    opt_list.append(np.mean(opt_list2))

mpl.plot(c1_range,opt_list)
mpl.show()

# c1 = .8
# c2 = .9
# x = p1_range - np.matrix(np.ones((p1_res, 1))*c1) # payoffs (price on columns)
#
# y = p2_range - np.matrix(np.ones((p2_res, 1))*c2) # payoffs (price on columns)
# # print(y)
# xx = np.multiply(f1_price_clear_matrix, x)
# yy = np.multiply(f2_price_clear_matrix, y)
# # print(xx)
# xxx = np.average(xx, axis=0) # find average payoff
# yyy = np.average(yy, axis=0)
# # print(xxx)
# f1_payoffs = np.multiply(f1_tie_break_matrix, xxx)
# f2_payoffs = np.multiply(f2_tie_break_matrix, yyy.T)
# print(f2_payoffs)
# print(f1_payoffs)
# print(br(f1_payoffs, f2_payoffs))