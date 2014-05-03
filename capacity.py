__author__ = 'msbcg452'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

target_resv = 1.15
fixed_cost = 77.4
eas_allowance = 28
cone = 77.4
eas_allowance = 28
eford = 0.057


IRM = target_resv - 1

fpr = (1 + IRM) * (1 - eford)


a = 9 / 5
b = 9 / 5
c = (np.log(9 / 4) / fixed_cost)
n_periods = 100
w_periods = n_periods - 5


def vrrc(fc_load):



    rel_req = fc_load * fpr

    p_a = max(cone, 1.5 * (cone - eas_allowance)) / 1-eford
    q_a = rel_req * (1 + IRM - 0.03) / (1 + IRM)

    p_b = 1.0 * (cone - eas_allowance) / (1 - eford)
    q_b = rel_req * (1 + IRM + 0.01) / (1 + IRM)

    p_c = 0.2 * (cone - eas_allowance) / (1 - eford)
    q_c = rel_req * (1 + IRM + 0.05) / (1 + IRM)

    def vrrc_curve(y):
        x = y*(1-FOR)
        if x <= q_a:
            return p_a
        if x <= q_b:
            return (x - q_a) * (p_b-p_a) / (q_b-q_a) + p_a
        if x <= q_c:
            return (x - q_b)*(p_c-p_b) / (q_c-q_b) + p_b
        if x > q_c:
            return 0

    return vrrc_curve, p_c, q_c/(1-FOR)

def gross_margin(reserve):
    return np.exp(14.88961658 + reserve/fpr * -11.5662)
    # return np.exp(21.79737 + reserve * 11.5662)



def utility(profit):
    u = a - b * np.exp(-c * profit)
    return u


def demand_curve(reserve):
    if reserve < target_resv:
        return 2 * fixed_cost - eas_allowance
    else:
        return 0


load_growth_avg = 0.017
weather_stdev = 0.01
err_wn = stats.norm(loc=0, scale=weather_stdev)
actual_stdev = 0.04
err_a = stats.norm(loc=0, scale=actual_stdev)

wn_load = np.zeros(n_periods)
ac_load = np.zeros(n_periods)
fc_load = np.zeros(n_periods)

wn_load[0] = 56.667
# ac_load[0] = wn_load[0] * (1+err_a.rvs())
# fc_load[0+4] = wn_load[0] * (1+load_growth_avg)**4

FOR = 0.057
beta = 0.07
max_cap = 0.084
price_cap = np.zeros(n_periods)
installed_cap = np.zeros(n_periods)
ac_resv = np.zeros(n_periods)
fc_resv = np.zeros(n_periods)
installed_cap[0] = 64.022
installed_cap[1:7] = np.array([installed_cap[0] * (1 + load_growth_avg) ** i for i in range(1, 7)])
fixed_cost = 69
gm = np.zeros(n_periods)
u = np.zeros(n_periods)
profit = np.zeros(n_periods)
weighted_util = np.zeros(n_periods)
rafp = np.zeros(n_periods)
cap_add = np.zeros(n_periods)
new_cap = np.zeros(n_periods)
weights = np.array([.5005 * (.5 ** (7 - i)) for i in range(0, 8)])

for i in range(0, 4):
    wn_load[i + 1] = wn_load[i] * (1 + load_growth_avg + err_wn.rvs())
    ac_load[i] = wn_load[i] * (1 + err_a.rvs())
    fc_load[i + 4] = wn_load[i] * (1 + load_growth_avg) ** 4

    ac_resv[i] = (1 - FOR) * installed_cap[i] / ac_load[i]
    fc_resv[i] = ac_resv[i]
    #fc_resv[i+4] = (1 - FOR) * installed_cap[i+4] / fc_load[i+4]

for i in range(3, w_periods):
    wn_load[i] = wn_load[i - 1] * (1 + load_growth_avg + err_wn.rvs())
    ac_load[i] = wn_load[i] * (1 + err_a.rvs())
    fc_load[i + 4] = wn_load[i] * (1 + load_growth_avg) ** 4

    ac_resv[i] = (1 - FOR) * installed_cap[i] / ac_load[i]


    fc_resv[i+1:i+4] = (1 - FOR) * installed_cap[i+1:i+4] / fc_load[i+1:i+4]
    print(gross_margin(ac_resv[i - 3:i+1]) + price_cap[i - 3:i+1])
    ac_profit = gross_margin(ac_resv[i - 3:i+1]) + price_cap[i - 3:i+1] - fixed_cost
    fc_profit = gross_margin(fc_resv[i+1:i + 4]) + price_cap[i+1:i + 4] - fixed_cost
    # print(fc_profit)
    # print(fc_resv[i:i+3])

    z, last_p, last_q = vrrc(fc_load[i+4])
    proj_res = fc_resv[i + 3]
    proj_price = z(fc_resv[i+3]*fc_load[i+4])
    proj_profit = gross_margin(proj_res) + proj_price - fixed_cost

    profit_slice = np.zeros(8)
    profit_slice[0:4] = ac_profit
    profit_slice[4:7] = fc_profit
    profit_slice[7] = proj_profit

    util_slice = utility(profit_slice)
    weighted_util[i + 4] = np.dot(util_slice, weights)
    rafp[i + 4] = -np.log((a - weighted_util[i + 4]) / b / c)

    new_cap[i + 4] = installed_cap[i + 3] * min(beta, max(0, load_growth_avg + (beta - load_growth_avg) *
                                                                     weighted_util[i + 4]))

    # capacity ratio calculation

    if new_cap[i+4] + installed_cap[i+3] > last_q:
        cap_add[i + 4] = max(0, last_q - installed_cap[i+3])
        price_cap[i+4] = last_p
    else:
        cap_add[i+4] = new_cap[i+4]
        price_cap[i+4] = z(cap_add[i+4] + installed_cap[i+3])
    installed_cap[i+4] = cap_add[i+4] + installed_cap[i+3]
    # # old capacity ratio calculation
    # cap_ratio = installed_cap[i + 2] / (fc_load[i + 3] * target_resv)
    # price_cap[i + 3] = 2*fixed_cost - eas_allowance if cap_ratio < 1 else 0
    # cap_add[i + 3] = max(0, min(fc_load[i + 3] * target_resv - installed_cap[i + 2], new_cap[i + 3]))
    # installed_cap[i + 3] = installed_cap[i + 2] + cap_add[i + 3]

    # hacked equilibrium calculation
    # cap_add[i + 3] = max(0, min(fc_load[i + 3] * target_resv - installed_cap[i + 2], new_cap[i + 3]))
    #
    # installed_cap[i + 3] = installed_cap[i + 2] + cap_add[i + 3]
    # price_cap[i + 3] = 2*fixed_cost-eas_allowance if cap_add[i + 3] > 0 else 0  # todo add demand curve

print(installed_cap)
print(fc_resv)
print(cap_add)
print(price_cap)

t = np.arange(0, w_periods)
# plt.plot(t, ac_load[0:w_periods], label="actual load")
# plt.plot(t, wn_load[0:w_periods], label="wn load")
# plt.plot(t, installed_cap[0:w_periods], label="installed cap")
#
# plt.legend()
# plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
ax1.plot(t, ac_load[0:w_periods], label="actual load")
ax1.plot(t, wn_load[0:w_periods], label="wn load")
ax1.plot(t, (1-FOR)*installed_cap[0:w_periods], label="installed cap")
ax1.set_title('loads')
ax1.legend()

ax2.plot(t, ac_resv[0:w_periods])
ax2.plot(t, fc_resv[0:w_periods])
ax2.axhline(fpr, 0, 1, ls='--', c='k')
ax2.set_title('reserves')

ax3.plot(t, new_cap[0:w_periods], label="new cap additions")
ax3.plot(t, cap_add[0:w_periods], label="cleared capacity")
ax3.legend()
ax3.set_title('capacity offered,cleared')

ax4.plot(t, weighted_util[0:w_periods])
plt.show()