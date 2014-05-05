__author__ = 'Emre'

# from ggplot import *
import matplotlib.pyplot as plt
target_resv = 1.15

cone = 77.4
eas_allowance = 28
eford = 0.057
IRM = target_resv - 1
import numpy as np
fpr = (1 + IRM) * (1 - eford)

def vrrc():
    rel_req = fpr

    p_a = max(cone, 1.5 * (cone - eas_allowance)) / 1-eford
    q_a = rel_req * (1 + IRM - 0.03) / (1 + IRM)

    p_b = 1.0 * (cone - eas_allowance) / (1 - eford)
    q_b = rel_req * (1 + IRM + 0.01) / (1 + IRM)

    p_c = 0.2 * (cone - eas_allowance) / (1 - eford)
    q_c = rel_req * (1 + IRM + 0.05) / (1 + IRM)

    def vrrc_curve(y):
        x = y
        if x <= q_a:
            return p_a
        if x <= q_b:
            return (x - q_a) * (p_b-p_a) / (q_b-q_a) + p_a
        if x <= q_c:
            return (x - q_b)*(p_c-p_b) / (q_c-q_b) + p_b
        if x > q_c:
            return 0

    return vrrc_curve, p_c, q_c/(1-eford)


z,p,q = vrrc()

t = np.linspace(.95, 1.15, 200)
y = [z(x) for x in t]

s = [2*cone-eas_allowance if x < fpr else 0 for x in t]
plt.plot(t-1, y, lw=3)
plt.plot(t-1 ,s, lw=3)
plt.xlabel('Unforced Reserve Margin',fontsize='large')
plt.ylabel('UCAP price $/MW-yr',fontsize='large')
# plt.title('Demand')
plt.xlim([0,.15])
plt.show()