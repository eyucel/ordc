__author__ = 'msbcg452'
import numpy as np
np.random.seed()
k = 4000000
n = 4
# generate random bids
bids = np.random.rand(k, n)
# p = np.random.rand(k,2)
# level 1 price
ps = .9
alpha = .1
# other prices
p = np.array([ps, ps-alpha, ps-2*alpha, ps-3*alpha])
print(p)
# set player 1 bids to .5
bids[:,0] = .65
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
ind = np.where(sorted_bid_owners == 0)
# print(ind)

# if you were in the cheapest bids list, so if you were cheapest, ind was 0
# and you would clear as long as a price cleared. return which price you received (0, 1, 2 or 3)
mask = np.where(num_clearing > ind[1], num_clearing, 0)
# print(mask)

# find where you received 2s
mask2 = mask == 2


# print(mask2)

# probability that you won 2s
print(np.mean(mask2))