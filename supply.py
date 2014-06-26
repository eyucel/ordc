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

    x0 = relreq*(1+irm-0.03)/(1+irm)
    y0 = max(1.5*(cone-eas),cone)/(1-eford)
    x1 = relreq*(1+irm+0.05)/(1+irm)
    y1 = (0.2*(cone-eas))/(1-eford)
    m = (y1-y0)/(x1-x0)

    if q <= x0:
        return y0
    if q < x1:
        return m*(q-x0)+y0
    else:
        return 0


init_supply = 105
size = .4
firms = 40

min_bid = 0.2
max_bid = 1.5
delta = max_bid - min_bid


bidrv = stats.uniform(min_bid, delta)
#bidrv2 = stats.uniform(min_bid, delta)




bids = bidrv.rvs(firms)
sorted_bid_owners = np.argsort(bids)
sorted_bids = np.sort(bids)


print(bids)
print(sorted_bids)
print(sorted_bid_owners)




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


# while vrr(qx) > sorted_bids[-i]:
#     i += 1



# if vrr(init_supply + 2*size) > sorted_bids[-i]:
#     pstar = vrr(qx)
#     qstar = qx
# else:
#     pstar = 0
#     qstar = 0
# print(pstar,qstar)


##### testing strategy
steps = 100
simuls = 5000
own_bids = np.linspace(min_bid, max_bid,steps)
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

avg_profit = np.mean(profit,1)


# print(cleared_count)
plt.plot(own_bids, avg_profit)
plt.figure()
plt.plot(own_bids, cleared_count)
plt.show()