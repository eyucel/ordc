__author__ = 'msbcg452'
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

cone = 1
eas = 0
netcone = 1-eas

irm = 0.15
eford = 0
peak = 100
fpr = (1+irm)*(1-eford)
relreq = fpr*peak
def vrr(q):

    # x0 = relreq*(1+irm-0.03)/(1+irm)
    # y0 = max(1.5*(cone-eas),cone)/(1-eford)
    # x1 = relreq*(1+irm+0.05)/(1+irm)
    # y1 = (0.2*(cone-eas))/(1-eford)
    # m = (y1-y0)/(x1-x0)

    # if q <= x0:
    #     return y0
    # if q < x1:
    #     return m*(q-x0)+y0
    # else:
    #     return 0

    p_a = max(cone, 1.5 * (cone - eas)) / 1-eford
    q_a = relreq * (1 + irm - 0.03) / (1 + irm)

    p_b = .8 * (cone - eas) / (1 - eford)
    q_b = relreq * (1 + irm + 0.01) / (1 + irm)

    p_c = 0 * (cone - eas) / (1 - eford)
    q_c = relreq * (1 + irm + 0.08) / (1 + irm)

    if q <= q_a:
        return p_a
    if q <= q_b:
        return (q - q_a) * (p_b-p_a) / (q_b-q_a) + p_a
    if q <= q_c:
        return (q - q_b)*(p_c-p_b) / (q_c-q_b) + p_b
    if q > q_c:
        return 0






init_supply = 110
size = .4
firms = 40

min_bid = 0.2
max_bid = 1.5
delta = max_bid - min_bid


bidrv = stats.uniform(min_bid, delta)
# bidrv = stats.norm(netcone, .5)
# bidrv = stats.expon(0,.5)
#bidrv2 = stats.uniform(min_bid, delta)
print(bidrv.mean())



bids = bidrv.rvs(firms)
sorted_bid_owners = np.argsort(bids)
sorted_bids = np.sort(bids)


print(bids)
print(sorted_bids)
print(sorted_bid_owners)

t = np.linspace(init_supply, init_supply + firms*size, 200)
vt = [vrr(x) for x in t]

capacity = [init_supply + size*i for i in range(1,firms+1)]
# plt.figure()
# plt.plot(capacity, sorted_bids)
# plt.plot(t,vt)
# plt.show()

for i in range(0,firms):
    qx = init_supply + (firms-i)*size
    print(qx)
    if vrr(qx) >= sorted_bids[-1-i]:
        # print(vrr(qx))
        # print(qx)
        cleared = qx
        break
    cleared = None

if not cleared:
    cleared = init_supply
print(cleared)



##### testing strategy
for tt in range(0, 80+1):
    init_supply = 108 + tt/10
    size = 0.4
    firms = 20
    min_bid = 0.5
    max_bid = 1.5
    delta = max_bid - min_bid



    steps = 100
    simuls = 1000
    own_bids = np.linspace(min_bid, max_bid, steps)
    bidrv = stats.uniform(min_bid, delta)
    #bidrv2 = stats.uniform(min_bid, delta)

    profit = np.zeros((steps, simuls))
    cleared_count = np.zeros(steps)
    j = 0
    for own_bid in own_bids:

        for s in range(0,simuls):
            bids = list(bidrv.rvs(firms-1))

            bids.append(own_bid)

            sorted_bid_owners = np.argsort(bids)
            sorted_bids = np.sort(bids)

            # print(sorted_bids)

            # print(bids)
            # print(sorted_bids)
            # print(sorted_bid_owners)


            for i in range(0, firms):
                qx = init_supply + (firms-i)*size
                #print(qx)
                if vrr(qx) >= sorted_bids[-1-i]:
                    # print(vrr(qx))
                    # print(qx)
                    cleared = qx

                    cleared_list = sorted_bid_owners[0:firms-i]
                    break
                cleared = None

            if not cleared:
                cleared = init_supply
                cleared_list = None
                pstar = 0

            else:
                pstar = vrr(cleared)
                if firms-1 in cleared_list:
                    cleared_count[j] += 1
                    profit[j,s] = pstar - netcone
        j+=1

    avg_profit = np.mean(profit, 1)


    # print(cleared_count)
    plt.figure()
    plt.plot(own_bids, avg_profit)
    plt.axis([min_bid, max_bid,-.3,.6])
    plt.title(format(init_supply,'>4'))
    plt.savefig(format(tt,'0=4')+'.png')
    # plt.show()


# plt.figure()
# plt.plot(own_bids, cleared_count)

#
#
# ##### testing two players
# firms = 2
# size = 1
# init_supply = 113
# steps = 100
# steps2 = 100
# min_bid = 0.5
# max_bid = 1.5
# own_bids = np.linspace(min_bid, max_bid, steps)
# other_bids = np.linspace(min_bid, max_bid, steps2)
#
# #bidrv2 = stats.uniform(min_bid, delta)
#
# profit = np.zeros((steps, steps2))
# cleared_count = np.zeros((steps, steps2))
# j = 0
#
# for own_bid in own_bids:
#     s = 0
#     for other_bid in other_bids:
#         bids = [own_bid] + [other_bid]*(firms-1)
#
#         sorted_bid_owners = np.argsort(bids)
#         sorted_bids = np.sort(bids)
#
#         # print(sorted_bids)
#
#         # print(bids)
#         # print(sorted_bids)
#         # print(sorted_bid_owners)
#
#
#         for i in range(0, firms):
#             qx = init_supply + (firms-i)*size
#             #print(qx)
#             if vrr(qx) >= sorted_bids[-1-i]:
#                 # print(vrr(qx))
#                 # print(qx)
#                 cleared = qx
#
#                 cleared_list = sorted_bid_owners[0:firms-i]
#                 break
#             cleared = None
#
#         if not cleared:
#             cleared = init_supply
#             cleared_list = None
#             pstar = 0
#
#         else:
#             pstar = vrr(cleared)
#             if firms-1 in cleared_list:
#                 cleared_count[j] += 1
#                 profit[j, s] = pstar - netcone
#         s+=1
#     j+=1
#
# print(profit)
#
# xx, yy = np.meshgrid(own_bids, other_bids)
# #plt.figure()
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xx,yy, profit, rstride=1, cstride=1, cmap=cm.coolwarm,
#         linewidth=0, antialiased=False)
#
# plt.show()
# pass


# plt.show()