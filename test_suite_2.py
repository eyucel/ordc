__author__ = 'Emre'
import numpy as np
from scipy import optimize
from scipy import stats


def linear_vrr(q,k):
    m = -1.5/10
    int = 1.5

    p =  m * q + (m*k+int)
    return np.maximum(p,0)


def fixed_simulate(draws, own_bf=1, opp_bf=1, k=0, size=0.5):



    min_bid = 0.5
    max_bid = 1.5
    delta = max_bid - min_bid


    j = 0
    optimal_bid_factor = own_bf
    opponent_bid_factor = opp_bf
    simuls, firms = draws.shape

    quantities = np.linspace(0, firms*size, firms, endpoint=False)
    prices = linear_vrr(quantities,k)
    prices = np.array([1,-10])
    # print(prices)
    # payout_history = np.zeros(simuls)
    # for s in range(0, simuls):
    #
    #     bids = list(opponent_bid_factor * draws[s,0:firms-1])
    #     own_cost = draws[s, -1]
    #     own_bid = optimal_bid_factor * own_cost
    #     bids.append(own_bid)
    #
    #     sorted_bid_owners = np.argsort(bids)
    #     sorted_bids = np.sort(bids)
    #
    #     # get price steps of all offers
    #
    #
    #
    #     cleared = sorted_bids < prices
    #     num_cleared = sum(cleared)
    #
    #     if (firms-1) in sorted_bid_owners[0:num_cleared]:
    #         payout = prices[num_cleared-1] - own_cost
    #     else:
    #         payout = 0
    #     payout_history[s] = payout

    bids = opponent_bid_factor*draws
    bids[:, firms-1] *= own_bf / opp_bf
    sorted_bid_owners = np.argsort(bids,axis=1)
    sorted_bids = np.sort(bids,axis=1)
    cleared = sorted_bids < prices
    num_cleared = np.sum(cleared,axis=1)

    ind = np.where(sorted_bid_owners == firms-1)
    mask = np.where(num_cleared > ind[1],1,0)

    payout_history = (prices[num_cleared-1]-draws[:, -1]) * mask
    # print(np.nonzero(num_cleared))
    # print(bids)
    # print(prices)
    # print(sorted_bid_owners)
    # print(sorted_bids)
    # print(cleared)
    # print(num_cleared)
    # print(ind)
    # print(mask)
    # print(payout_history)
    # input()
    exp_payout = np.mean(payout_history)
    return exp_payout

def fixed_func(bf, draws, k, size):

    val = -fixed_simulate(draws, own_bf=bf, opp_bf=bf,k=k, size=size)
    return val


if __name__=="__main__":
    min_bid = .5
    max_bid = 1.5
    delta = max_bid-min_bid
    trials = 10
    size = 1
    competitors = [2]
    for c in competitors:
        print("c=",c)
        # for k in range(7):
            # print("k=",k)
        for k in [7/3]:

            z_hist = np.zeros((trials))
            bidrv = stats.uniform(min_bid, delta)
            for t in range(trials):
                d = bidrv.rvs((200000, c))
                # print(simulate(1000,.5))

                z = 1
                tol = 1e-3
                bracket = .4
                bracket_reduc = .75
                lb = (1-bracket) * z
                rb = (1+bracket) * z
                z = optimize.fminbound(fixed_func, lb, rb, args=(d, k, size), disp=1)
                    # bracket *= bracket_reduc
                # z = optimize.basinhopping(func,1,disp=True)
                print(z)
                z_hist[t] = z

                # print("Trial:", t)

            print(np.mean(z_hist))
