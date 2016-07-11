import mpmath as mp
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import beta


mp.dps = 1000
mp.pretty = True




n = 4


j = n-1
# n here is other number of players aka N-1
# k is # number of winners
k = 2

H = lambda c, w: (-(1-c)**(j-k) * c**(k-1) * (w-1) * (1-2*c + w))/(2 * mp.beta(k, j+1-k) * (1 - mp.betainc(k, j+1-k, 0, c, regularized=True)))
G = lambda c, w: (-(1-c)**(j-k) * c**(k-1) * (w-1) * (1-2*c + w))/(2 * beta(k, j+1-k) * (1 - mp.betainc(k, j+1-k, 0, c, regularized=True)))
num = lambda c, w: -(1-c)**(j-k) * c**(k-1) * (w-1) * (1-2*c + w)
denom = lambda c,w: 2 * mp.beta(k, j+1-k) * (1 - mp.betainc(k, j+1-k, 0, c, regularized=True))

func = lambda c, w: (2**(-1 - n) * (2 - c)**(-k + n) * c**(-1 +  k) * (-4 + 4 * c - 2 * c * w + w**2))  \
     / (mp.beta(k, 1 - k + n)* (-1 + mp.betainc( k, 1 - k + n,0, c/2) * (c - w)))

print(H(.2,.5))
print(G(.2,.5))
c_end = mp.mpf('0')
c_0 = mp.mpf('1.995')
w_0 = mp.mpf('1.99')

w_c = [w_0]
cs = [c_0]

nt = 1001
incr = mp.mpf('0.001')
print(incr)
c = c_0
i = 0
while c > 0:

    w = w_c[i]
    c = cs[i]
    # n = num(c,w)
    # d = denom(c,w)
    # print(type(d))
    # rhs = n/d

    rhs=func(c,w)
    print(rhs)
    w_c.append(w - incr * rhs )
    cs.append(cs[i]-incr)
    i+=1
    print(i)

print(w_c)
plt.plot(cs,w_c)
plt.show()







# plt.show()

