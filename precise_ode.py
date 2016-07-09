import mpmath as mp
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import beta

mp.dps = 200
mp.pretty = True




n = 10


j = n-1
# n here is other number of players aka N-1
# k is # number of winners
k = 2

H = lambda c, w: (-(1-c)**(j-k) * c**(k-1) * (w-1) * (1-2*c + w))/(2 * mp.beta(k, j+1-k) * (1 - mp.betainc(k, j+1-k, 0, c, regularized=True)))
G = lambda c, w: (-(1-c)**(j-k) * c**(k-1) * (w-1) * (1-2*c + w))/(2 * beta(k, j+1-k) * (1 - mp.betainc(k, j+1-k, 0, c, regularized=True)))
num = lambda c, w: -(1-c)**(j-k) * c**(k-1) * (w-1) * (1-2*c + w)
denom = lambda c,w: 2 * mp.beta(k, j+1-k) * (1 - mp.betainc(k, j+1-k, 0, c, regularized=True))
print(H(.2,.5))
print(G(.2,.5))
c_end = mp.mpf('0.01')
c_0 = mp.mpf('.999')
w_0 = mp.mpf('.997')

w_c = [w_0]
cs = [c_0]

nt = 200
incr = (c_0 - c_end)/nt
for i in range(0,nt):

    w = w_c[i]
    c = cs[i]
    n = num(c,w)
    d = denom(c,w)
    rhs = n/d
    # print(rhs)
    w_c.append(w-incr * rhs )
    cs.append(cs[i]-incr)
    print(i)
print(w_c)
plt.plot(cs,w_c)
plt.show()







# plt.show()

