import numpy as np

aucpro = 1
n = 3 # number of h-types
big_J = 2 # taylor series order expansion
nt = 200 # of grid points

k = np.zeros(n)
u = np.zeros(n)
t = np.zeros(nt)
Fres = np.zeros(n)
a = np.zeros((n, big_J))
b = np.zeros((n, big_J))
c = np.zeros((n, big_J))
d = np.zeros((n, big_J))
p = np.zeros((n, big_J))
q = np.zeros((n, big_J))
l = np.zeros((n, nt))
ll = np.zeros((n, nt))
lpl = np.zeros((n, nt))
check = np.zeros(n)
bids = np.zeros((n, nt))
biga = np.zeros((n,n))
p0 = np.zeros((n, nt))
bigb = np.zeros((n,))
a1 = np.zeros((n, n))
a2 = np.zeros((n, n))
b1 = np.zeros((n, n))
b2 = np.zeros((n, 1))
c1 = np.zeros((big_J, n))
c2 = np.zeros((n, n))
ta = np.zeros((n, nt))
tmean = np.zeros(n)
tstd = np.zeros(n)
lf = np.zeros(n)
uf = np.zeros(n)
lff = np.zeros(n)
uff = np.zeros(n)
pf = np.zeros(n)
cf = np.zeros(n)
pff = np.zeros(n)
cff = np.zeros(n)
pf1 = np.zeros(n)
cf1 = np.zeros(n)
pff1 = np.zeros(n)
cff1 = np.zeros(n)

k = np.array([1, 1, 1])

lv = 0
uv = 5
res = 0
n_cdf = 3 # three b-cdfs
ccl = np.zeros(n_cdf)
ccu = np.zeros(n_cdf)
i_cdf = np.zeros(n_cdf)
musd = np.zeros((n_cdf, 2))
qcdf = np.zeros((n, n_cdf))

i_cdf = np.array([1, 1, 1]) # three weibull
musd[0, :] = np.array([1, 1])
musd[1, :] = np.array([2, 1])
musd[2, :] = np.array([3.39, 2.2])
aa = musd[:, 0]
bb = musd[:, 1]

qcdf[0, :] = np.array([1, 0, 0])
qcdf[1, :] = np.array([0, 1, 0])
qcdf[2, :] = np.array([0, 0, 1])
