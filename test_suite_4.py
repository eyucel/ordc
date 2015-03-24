__author__ = 'msbcg452'
import numpy as np
from scipy.special import lambertw

def bidf(c):
    num = alpha*alpha * -lambertw(-np.exp(alpha*alpha * (4*c-5) + 4*alpha*(c-1)+c-1)) - alpha*alpha + 2 * alpha * c + c
    denom = 2*alpha+1
    w = num/denom
    y = np.where(w<0, 0, w)
    return y
n = 3
# k = 2

ps = .9
alpha = .1
# other prices
p = np.array([ps, ps-alpha, ps-2*alpha])
p = np.where(p<0,0,p)
f = None
# f = bidf
M = 1000000
# own_idx = np.random.random_integers(0,n-1,size=(M,1))
# generate random bids
costs = np.random.rand(M, n)
# costs[:, 0] = .65
sorted_owners = np.argsort(costs, axis=1)
sorted_costs = np.sort(costs, axis=1)
# sorted_costs[sorted_costs <.001] = .001
# sorted_costs[sorted_costs >.998] = .998
# print(costs)
if f is None:
    sorted_bids = sorted_costs
else:
    sorted_bids = f(sorted_costs)
# p = np.random.rand(k,2)
# level 1 price

# p = np.array(k*[ps]+(n-k)*[-1])

# print(p)
# set player 1 bids to something fixed.

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

# whether or not you were in the cheapest bids list, so if you were cheapest, ind was 0
# and you would clear as long as a price cleared. return which price you received (1, 2, ..., k)
mask = np.where(num_clearing > ind[1], num_clearing, 0)
# print(mask)
# print(mask)
# find where you received 2s
mask2 = mask == 2
maskany = mask > 0
# print(np.mean(maskany))
received_price = [p[x-1] if x > 0 else 0 for x in mask]
print(np.mean((received_price-costs[ind])))
# print(sum(mask)/M)r

# probability that you won 2s
# print(np.mean(mask2))




f = bidf
sorted_owners = np.argsort(costs, axis=1)
sorted_costs = np.sort(costs, axis=1)
if f is None:
    sorted_bids = sorted_costs
else:
    sorted_bids = f(sorted_costs)
# p = np.random.rand(k,2)
# level 1 price

# p = np.array(k*[ps]+(n-k)*[-1])

# print(p)
# set player 1 bids to something fixed.

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
# and you would clear as long as a price cleared. return which price you received (1, 2, ..., k)
mask = np.where(num_clearing > ind[1], num_clearing, 0)
# print(mask)
# print(mask)
# find where you received 2s
mask2 = mask == 2
maskany = mask > 0
# print(np.mean(maskany))
received_price = [p[x-1] if x > 0 else 0 for x in mask]
print(np.mean((received_price-costs[ind])))