import numpy as np
import scipy.interpolate
import scipy.stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class storage:
    def __init__(self):
        pass

npt = 1001
ngrid = 60
ko = 6
nko = npt + ko
nq = 200
told = 1e-3
ftol = 1e-6


aucpro = 1
n = 3 # number of h-types
big_J = 2 # taylor series order expansion
nt = 200 # of grid points


bad_form = storage()

k = np.zeros(n)
u = np.zeros(n)
t = np.zeros(nt+1)
Fres = np.zeros(n)
a = np.zeros((n, big_J+1))
b = np.zeros((n, big_J+1))
c = np.zeros((n, big_J+1))
d = np.zeros((n, big_J+1))
p = np.zeros((n, big_J+1))
q = np.zeros((n, big_J+1))
l = np.zeros((n, nt+1))
ll = np.zeros((n, nt+1))
lpl = np.zeros((n, nt+1))
check = np.zeros(n)
bids = np.zeros((n, nt+1))
biga = np.zeros((n, n))
p0 = np.zeros((n, nt+1))
bigb = np.zeros((n, 1))
a1 = np.zeros((n, n))
a2 = np.zeros((n, n))
b1 = np.zeros((n, n))
b2 = np.zeros((n, 1))
c1 = np.zeros((big_J, n))
c2 = np.zeros((n, n))
ta = np.zeros((n, nt+1))
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
print(musd)
dist_list = [scipy.stats.weibull_min(musd[i, 0], scale=musd[i, 1]) for i in range(0, n)]

# p1_dist = scipy.stats.weibull_min(musd[0, 0], scale=musd[0, 1])
# p2_dist = scipy.stats.weibull_min(musd[1, 0], scale=musd[1, 1])
# p3_dist = scipy.stats.weibull_min(musd[2, 0], scale=musd[2, 1])


qcdf[0, :] = np.array([1, 0, 0])
qcdf[1, :] = np.array([0, 1, 0])
qcdf[2, :] = np.array([0, 0, 1])

# equal spaced grid for inverse cdf
h = np.linspace(0, 1, npt)
# equal spaced grid for cdf
h1 = np.linspace(lv, uv, npt)

bcf = np.zeros((n, npt))
pcf = np.zeros((n, ko, npt))
br = np.zeros((n, npt))
bcf1 = np.zeros((n, npt))
pcf1 = np.zeros((n, ko, npt))
br1 = np.zeros((n, npt))
# mod_cdf = lambda v: (p1_dist.cdf(v)-p1_dist.cdf(lv))/(p1_dist.cdf(uv)-p1_dist.cdf(lv))
mod_cdf_list = [lambda v, i=i: (dist_list[i].cdf(v)-dist_list[i].cdf(lv))/(dist_list[i].cdf(uv)-dist_list[i].cdf(lv)) for i in range(0, n)]
# mod_ppf = lambda u: p1_dist.ppf(p1_dist.cdf(lv) + u * (p1_dist.cdf(uv)-p1_dist.cdf(lv)))
mod_ppf_list = [lambda u, i=i: dist_list[i].ppf(dist_list[i].cdf(lv) + u * (dist_list[i].cdf(uv)-dist_list[i].cdf(lv))) for i in range(0, n)]
# p1_spl = scipy.interpolate.splrep(h, mod_ppf(h), s=0, k=5)
# p1_spll = scipy.interpolate.splmake(h, mod_ppf(h), order=6)
# p1_pp = scipy.interpolate.PPoly.from_spline(p1_spl)
# p1_ppp = scipy.interpolate.PPoly.from_spline(p1_spll)
# p1_dppp = p1_ppp.derivative()
spl_list = [scipy.interpolate.splmake(h, mod_ppf_list[i](h), order=6) for i in range(0, n)]
cdf_list = [scipy.interpolate.splmake(h1, mod_cdf_list[i](h1), order=6) for i in range(0, n)]
pp_list = [scipy.interpolate.PPoly.from_spline(spl_list[i]) for i in range(0, n)]
ppc_list = [scipy.interpolate.PPoly.from_spline(cdf_list[i]) for i in range(0, n)]

print((dist_list[0].cdf(2)-dist_list[0].cdf(lv))/(dist_list[0].cdf(uv)-dist_list[0].cdf(lv)))
print((dist_list[1].cdf(2)-dist_list[1].cdf(lv))/(dist_list[1].cdf(uv)-dist_list[1].cdf(lv)))
print((dist_list[2].cdf(2)-dist_list[2].cdf(lv))/(dist_list[2].cdf(uv)-dist_list[2].cdf(lv)))
print(mod_cdf_list[0](2))
print(mod_cdf_list[1](2))
print(mod_cdf_list[2](2))
# print(p1_spl)
# print(p1_spll)
# print(p1_interp(np.linspace(0,1,50)))
# print(p1_interp(h))
# p1_spl = scipy.interpolate.splmake(h, p1_dist.ppf(h), order=3) # create spline interpolation
# print(h)
# print(p1_dist.ppf(h))
# print(scipy.interpolate.spleval(p1_spl, h))
# p1_pp = scipy.interpolate.PPoly.from_spline(p1_spl)
# print(p1_pp(h))
# # p1_pp = scipy.interpolate.spltopp(*p1_spl) # create piecewise polynomial of spl
# print(p1_pp)
# print(p1_spl)

# p2_interp = scipy.interpolate.interp1d(h, p2_dist.ppf(h), kind=6)
# p3_interp = scipy.interpolate.interp1d(h, p3_dist.ppf(h), kind=6)

# p2_interp = scipy.interpolate.PiecewisePolynomial(h, p2_dist.ppf(h), orders=ko)
# p3_interp = scipy.interpolate.PiecewisePolynomial(h, p3_dist.ppf(h), orders=ko)

bigN = np.sum(k)
sumk = 1.0/(bigN-1)

t_grid = np.linspace(res, uv, ngrid)

obj = np.zeros(ngrid)
obj[0] = 1000
obj[-1] = 1000
# f_icdf = lambda x: [1-pp_list[i](x, nu=0) for i in range(0, n)]
f_cdf1 = lambda x: [ppc_list[i](x, nu=0) for i in range(0, n)]
f_pdf1 = lambda x: [ppc_list[i](x, nu=1) for i in range(0, n)]




def concat(i, f0, g0):
    # i += 1
    # i = 1
    # print(i)
    # print(f0, g0)
    if i == 0:
        aa = f0[0]
    else:
        qq = np.zeros((i+1, i+1))
        qq[0, 0] = 1
        for d in range(1, i+1):
            for ll in range(1, d+1):
                for j in range(1, d+1-ll+1):
                    # print(ll,d, j+1, ll-1, d-j)
                    qq[ll, d] = qq[ll, d] + g0[j] * qq[ll-1, d-j]


        aa = np.dot(f0[1:i+1], qq[1:i+1, i])
    # print('exiting concat')
    return aa


def asym_recursion(tt):
    t = np.linspace(tt, res, nt+1)
    inc = (tt-res)/(nt)
    cdfres = f_cdf1(res)
    # print(cdfres)
    t[0] = res
    p0 = np.zeros((n, nt+1))
    l = np.zeros((n, nt+1))
    lpl = np.zeros((n, nt+1))
    for i in range(0, n):
        bids[i, :] = t
    lpl[:, -1] = np.array(f_pdf1(uv))*bigN/(bigN-1)
    l[:, -1] = 1

    obj1 = 0
    check = np.zeros(3)
    m = nt
    while(m >= 1):
        a = np.zeros((n, nt+1))
        a[:, 0] = l[:, m]
        # print(a[:, 0])
        for i in range(n):
            for j in range(big_J+1):
                fc = np.math.factorial(j)

                d[i, j] = (-1)**j * pp_list[i](1-a[i, 0], nu=j)/fc
        # print(d)
        # initialize other taylor series coefficients
        p[:, 0] = d[:, 0] - t[m]
        bigb[:, 0] = -1
        for i in range(0, n):
            a1[i, i] = 1/p[i, 0]
            a2[:, i] = k[i]/p[i, 0]
        biga[:, :] = a1 - a2 * sumk
        b2 = np.dot(biga, bigb)
        b[:, 0] = b2[:, 0]

        # need to store p0 for revenue calculations
        p0[:, m] = p[:, 0]

        # recursion to calculate a(:,i),i=1,...,bigJ
        for i in range(1, big_J+1):
            # calculate a(:,i)
            for j in range(0, i):
                a[:, i] = a[:, i] + a[:, j]*b[:, i-j-1]
            a[:, i] = a[:, i]/np.float64(1)

            # calculate p(:,i)

            for j in range(0, n):
                p[j, i] = concat(i, d[j, :], a[j, :])

            if i == 1:
                p[:, i] = p[:, i] - 1

            # calculate RHS of main equation
            for j in range(0, i):
                # print('is neg', i-j-1)
                c1[j, :] = b[:, i-j-1]

            c2 = np.dot(p[:, 1:i+1], c1[0:i, :])

            for j in range(0, n):
                bigb[j, 0] = np.sum(b1[j, :] * c2[j, :])

            # calculate new b
            b2 = np.dot(biga, bigb)
            b[:, i] = b2[:, 0]


        m -= 1
        # calculate new values of l and inverse bids, and lpl
        for i in range(0, big_J+1):

            l[:, m] += a[:, i] * ((-inc)**i)
            lpl[:, m] += b[:, i] * ((-inc)**i)
            bids[:, m+1] += p[:, i] * ((-inc)**i)
        # print(a[:, i], inc)
        check += np.where(l[:, m] - cdfres < 0, 1, 0)
        # print(l[:, m])
        # print(l[:, m+1])
        check += np.where(l[:, m] > 1, 1, 0)
        check += np.where(l[:, m] > l[:, m+1], 1, 0)
        # print(check)
        # print(l)
        # print(lpl)
        if np.sum(check) > 0:
            obj1 = 1000
            # print(m)
            m = 0
            # print("check exit")
        elif sum(check)==0 and m<=2:
            obj1 += np.sqrt(np.sum(p[:, 0]**2))

    bad_form.bids = bids
    return obj1






for i in range(1, ngrid):
    obj[i] = asym_recursion(t_grid[i])
    print(obj[i])
    if obj[i] > obj[i-1]:
        print("exiting b")
        break
tstar = t_grid[i-1]
ginc = (uv-res)/(ngrid-1)
xint = ginc
dogrid = False
print(obj)

res = minimize(asym_recursion, tstar, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print(bad_form.bids[:, :-1])
plt.plot(bad_form.bids[:, :-1].T)
plt.show()