__author__ = 'msbcg452'


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

target_resv = 1
fixed_cost = 69
a = 9/5
b = 9/5
c = (np.log(9/4)/fixed_cost)


def gross_margin(reserve):
    return np.exp(21.79737 + reserve*11.5662)


def utility(profit):
    u = a - b * np.exp(-c * profit)
    return u


def demand_curve(reserve):
    if reserve < target_resv:
        return 120
    else:
        return 0


load_growth_avg = 0.017
weather_stdev = 0.01
err_wn = stats.norm(loc=0, scale=weather_stdev)
actual_stdev = 0.04
err_a = stats.norm(loc=0, scale=actual_stdev)

wn_load = np.zeros(30)
ac_load = np.zeros(30)
fc_load = np.zeros(30)

wn_load[0] = 56.667
# ac_load[0] = wn_load[0] * (1+err_a.rvs())
# fc_load[0+4] = wn_load[0] * (1+load_growth_avg)**4

FOR = 0.057
beta = 0.07
max_cap = 0.084
price_cap = np.zeros(30)
installed_cap = np.zeros(30)
ac_res = np.zeros(30)
fc_res = np.zeros(30)
installed_cap[0] = 64.022
installed_cap[1:7] = np.array([installed_cap[0]*(1+load_growth_avg)**i for i in range(1, 7)])
fixed_cost = 69
gm = np.zeros(30)
u = np.zeros(30)
profit = np.zeros(30)
weighted_util = np.zeros(30)
rafp = np.zeros(30)
cap_add = np.zeros(30)
new_cap = np.zeros(30)
weights = np.array([.5005 * (.5**(7-i)) for i in range(0, 8)])

for i in range(0, 4):
    wn_load[i+1] = wn_load[i] * (1 + load_growth_avg + err_wn.rvs())
    ac_load[i] = wn_load[i] * (1 + err_a.rvs())
    fc_load[i+4] = wn_load[i] * (1 + load_growth_avg)**4

    ac_res[i] = (1-FOR) * installed_cap[i]/ac_load[i]
    fc_res[i] = ac_res[i]

for i in range(4, 25):
    wn_load[i] = wn_load[i-1] * (1 + load_growth_avg + err_wn.rvs())
    ac_load[i] = wn_load[i] * (1 + err_a.rvs())
    fc_load[i+4] = wn_load[i] * (1 + load_growth_avg)**4

    ac_res[i] = (1-FOR) * installed_cap[i]/ac_load[i]
    fc_res[i] = (1-FOR) * installed_cap[i]/fc_load[i]

    ac_profit = gross_margin(ac_res[i-4:i]) + price_cap[i-4:i] - fixed_cost
    fc_profit = gross_margin(fc_res[i:i+3]) + price_cap[i:i+3] - fixed_cost

    proj_res = fc_res[i+2]
    proj_price = demand_curve(proj_res)
    proj_profit = gross_margin(proj_res) + proj_price - fixed_cost

    profit_slice = np.zeros(8)
    profit_slice[0:4] = ac_profit
    profit_slice[4:7] = fc_profit
    profit_slice[7] = proj_profit
    util_slice = utility(profit_slice)
    weighted_util[i+3] = np.dot(util_slice,weights)
    rafp[i+3] = -np.log((a-weighted_util[i+3])/b/c)

    new_cap[i+3] = installed_cap[i+2] * np.amin(beta, np.amax(0, load_growth_avg+(beta-load_growth_avg)*weighted_util[i+3]))
    cap_add[i+3] = max(0, min(fc_load[i+3]*target_resv - installed_cap[i+2], new_cap[i+3]))

    installed_cap[i+3] = installed_cap[i+2]+cap_add[i+3]
    price_cap[i+3] = 120 if cap_add[i+3]>0 else 0


print(installed_cap)
print(ac_res)
print(cap_add)
t = np.arange(0,25)
plt.plot(t, ac_load[0:25], label="actual load")
plt.plot(t, wn_load[0:25], label="wn load")
plt.plot(t, installed_cap[0:25], label="installed cap")
plt.legend()
plt.show()