__author__ = 'msbcg452'

__author__ = 'msbcg452'

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def task(x):
    return np.random.random(x)

def linear_vrr(q):
    m = -1.5/10
    int = 1.5
    k  = 3
    p =  m * q + (m*k+int)
    return np.maximum(p,0)



def simulate(simuls,own_bf=1,opp_bf=1):

    size = 0.4
    firms = 2
    min_bid = 0.5
    max_bid = 1.5
    delta = max_bid - min_bid



    steps = 1
    bidrv = stats.uniform(min_bid, delta)


    j = 0
    optimal_bid_factor = own_bf
    opponent_bid_factor = opp_bf


    quantities = np.linspace(0, firms*size, firms, endpoint=False)
    prices = linear_vrr(quantities)
    # print(prices)
    payout_history = np.zeros(simuls)
    np.random.seed()
    for s in range(0, simuls):

        bids = list(opponent_bid_factor * bidrv.rvs(firms-1) )
        own_cost = bidrv.rvs(1)
        own_bid = optimal_bid_factor * own_cost
        bids.append(own_bid)

        sorted_bid_owners = np.argsort(bids)
        sorted_bids = np.sort(bids)

        # get price steps of all offers



        cleared = sorted_bids < prices
        num_cleared = sum(cleared)

        if (firms-1) in sorted_bid_owners[0:num_cleared]:
            payout = prices[num_cleared-1] - own_cost
        else:
            payout = 0
        payout_history[s] = payout
        exp_payout = np.mean(payout_history)
    return exp_payout

def fixed_simulate(draws,own_bf=1,opp_bf=1):

    size = 0.4

    min_bid = 0.5
    max_bid = 1.5
    delta = max_bid - min_bid


    j = 0
    optimal_bid_factor = own_bf
    opponent_bid_factor = opp_bf
    simuls, firms = draws.shape

    quantities = np.linspace(0, firms*size, firms, endpoint=False)
    prices = linear_vrr(quantities)
    # print(prices)
    payout_history = np.zeros(simuls)
    for s in range(0, simuls):

        bids = list(opponent_bid_factor * draws[s,0:firms-1])
        own_cost = draws[s, -1]
        own_bid = optimal_bid_factor * own_cost
        bids.append(own_bid)

        sorted_bid_owners = np.argsort(bids)
        sorted_bids = np.sort(bids)

        # get price steps of all offers



        cleared = sorted_bids < prices
        num_cleared = sum(cleared)

        if (firms-1) in sorted_bid_owners[0:num_cleared]:
            payout = prices[num_cleared-1] - own_cost
        else:
            payout = 0
        payout_history[s] = payout
    exp_payout = np.mean(payout_history)
    return exp_payout

if __name__ == "__main__":
    min_bid = 0.5
    max_bid = 1.5
    delta = max_bid - min_bid


    steps = 1
    bidrv = stats.uniform(min_bid, delta)
    np.random.seed(5)
    draws = bidrv.rvs((10000,2))
    print(fixed_simulate(draws, 1, 1))
