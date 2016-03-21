
import numpy as np
import scipy.interpolate
import scipy.stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import starmap
import BSpline
import scipy as sp

class storage:
    def __init__(self):
        pass

left = 0.05
right = 1
step_count = 100
rs = np.linspace(left, right, step_count)
all_bids = []
all_tstars = []

for res in rs:



    npt = 1001
    ngrid = 60
    ko = 6
    nko = npt + ko
    nq = 200
    told = 1e-3
    ftol = 1e-6


    aucpro = 1
    n = 2 # number of h-types
    big_J = 2# taylor series order expansion
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

    # k = np.array([1, 1])
    # k = np.array([1, 1, 1])
    k = np.ones(n)

    lv = 0
    uv = 1

    n_cdf = 2 # three b-cdfs
    ccl = np.zeros(n_cdf)
    ccu = np.zeros(n_cdf)
    i_cdf = np.zeros(n_cdf)
    musd = np.zeros((n_cdf, 2))
    qcdf = np.zeros((n, n_cdf))

    i_cdf = np.array([1, 1, 1]) # three weibull
    musd[0, :] = np.array([1, 1])
    musd[1, :] = np.array([2, 1])
    try:
        musd[2, :] = np.array([3.39, 2.2])
    except:
        pass
    aa = musd[:, 0]
    bb = musd[:, 1]
    # print(musd)
    dist_list = [scipy.stats.weibull_min(musd[i, 1], scale=musd[i, 0]) for i in range(0, n)]
    dist_list = [scipy.stats.uniform(loc=0, scale=1) for i in range(0,n)]

    # p1_dist = scipy.stats.weibull_min(musd[0, 0], scale=musd[0, 1])
    # p2_dist = scipy.stats.weibull_min(musd[1, 0], scale=musd[1, 1])
    # p3_dist = scipy.stats.weibull_min(musd[2, 0], scale=musd[2, 1])


    # qcdf[0, :] = np.array([1, 0, 0])
    # qcdf[1, :] = np.array([0, 1, 0])
    # qcdf[2, :] = np.array([0, 0, 1])

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
    # spl_list = [scipy.interpolate.splmake(h, mod_ppf_list[i](h), order=6) for i in range(0, n)]
    # cdf_list = [scipy.interpolate.splmake(h1, mod_cdf_list[i](h1), order=6) for i in range(0, n)]
    # h = [0, 1, 1, 3, 4, 6, 6, 6]
    # h = list(range(0,6))
    # ko = 4
    # npt = 6
    # print(h)
    # print(ko-ko/2)
    # knots = np.zeros(npt+ko)
    # knots[0:ko] = h[0]
    # knots[ko:npt] = h[ko-ko//2: npt-ko//2]
    # knots[npt:]= h[-1]+told
    # # print(len(knots))
    # basis = BSpline.Bspline(knots, 6-1)
    # print([basis(i) for i in h])
    # zzz = np.array([basis(i) for i in h])
    # print(zzz)
    #
    # # print(basis(.5))
    # print(basis(4))
    # print(basis(5))
    # qsz = lambda x: np.array(x)**3
    # coefs = [sp.linalg.solve(zzz, mod_ppf_list[i](h)) for i in range(0,n)]
    # plt.plot(h, ([np.dot(qsz(h), basis(i)) for i in h]))
    # plt.plot(h, mod_ppf_list[0](h))
    # plt.show()
    # basis(h)
    # basis.plot()

    spl_list = [scipy.interpolate.splrep(h, 1-mod_ppf_list[i](h), k=5) for i in range(0, n)]
    cdf_list = [scipy.interpolate.splrep(h1, 1-mod_cdf_list[i](h1), k=5) for i in range(0, n)]
    # pp_list = [scipy.interpolate.PPoly.from_spline((knots, coefs[i], 5),extrapolate=None) for i in range(0,n)]
    pp_list = [scipy.interpolate.PPoly.from_spline(spl_list[i], extrapolate=False) for i in range(0, n)]
    ppc_list = [scipy.interpolate.PPoly.from_spline(cdf_list[i], extrapolate=False) for i in range(0, n)]
    # plt.plot(h,dist_list[0].cdf(h))
    # plt.plot(h,mod_cdf_list[0](h))
    # plt.figure()
    # plt.plot(h,pp_list[2](h, nu=0))
    # plt.figure()
    # plt.plot(h,pp_list[2](h, nu=1))
    # plt.figure()
    # plt.plot(h,pp_list[2](h, nu=2))
    # plt.show()
    # print(h)
    # print((dist_list[0].cdf(2)-dist_list[0].cdf(lv))/(dist_list[0].cdf(uv)-dist_list[0].cdf(lv)))
    # print((dist_list[1].cdf(2)-dist_list[1].cdf(lv))/(dist_list[1].cdf(uv)-dist_list[1].cdf(lv)))
    # print((dist_list[2].cdf(2)-dist_list[2].cdf(lv))/(dist_list[2].cdf(uv)-dist_list[2].cdf(lv)))
    # print(mod_cdf_list[0](2))
    # print(mod_ppf_list[0](0))
    # print(ppc_list[0](0))
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


    t_grid = np.linspace(lv, res, ngrid)

    obj = np.zeros(ngrid)
    obj[0] = 1000
    obj[-1] = 1000
    # f_icdf = lambda x: [1-pp_list[i](x, nu=0) for i in range(0, n)]
    f_cdf1 = lambda x: [1-ppc_list[i](x, nu=0) for i in range(0, n)]
    f_pdf1 = lambda x: [-ppc_list[i](x, nu=1) for i in range(0, n)]


    # print(f_pdf1(2))
    # print([dist_list[i].pdf(2) for i in range(0,n)])
    # print(np.array(f_pdf1(uv)))

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
            # if aa > 10000:
                # print(f0[1:i+1], qq[1:i+1, i])
        # print('exiting concat')
        return aa


    def asym_precursion(tt):
        t = np.linspace(tt, res, nt+1)
        t[nt] = res
        inc = (res-tt)/nt
        cdfres = 1-np.array(f_cdf1(res))
        # print(cdfres)
        t[0] = tt
        p0 = np.zeros((n, nt+1))
        l = np.zeros((n, nt+1))
        lpl = np.zeros((n, nt+1))
        for i in range(0, n):
            bids[i, :] = t
        lpl[:, 0] = -np.array(f_pdf1(uv))*bigN/(bigN-1)
        l[:, 0] = 1

        obj1 = 0
        check = np.zeros(n)
        m = 0
        while(m <= nt-1):
            a = np.zeros((n, big_J+1))
            a[:, 0] = l[:, m]
            # print(a[:,0])
            for i in range(n):
                for j in range(big_J+1):
                    fc = np.math.factorial(j)

                    # d[i, j] = (1)**j * pp_list[i](1-a[i, 0], nu=j)/fc
                    d[i, j] = ((1.0)**j)*pp_list[i](a[i, 0], nu=j)/fc
                    # print(i,j, pp_list[i](1-a[i, 0], nu=j ), t[m])

            # initialize other taylor series coefficients
            p[:, 0] = d[:, 0] - t[m]
            # print(d[:, 0])
            # print(p[:, 0])
            bigb[:, 0] = 1
            a1 = np.zeros((n, n))
            for i in range(0, n):
                a1[i, i] = 1/p[i, 0]
                a2[:, i] = k[i]/p[i, 0]
                # print(p[i, 0], k[i])
            biga[:, :] = a1 - a2 * sumk
            b2 = np.dot(biga, bigb)
            # print(biga, bigb)
            b[:, 0] = b2[:, 0]
            # print('b', b)
            # need to store p0 for revenue calculations
            p0[:, m] = p[:, 0]

            # recursion to calculate a(:,i),i=1,...,bigJ
            for i in range(1, big_J+1):
                # calculate a(:,i)
                for j in range(0, i):
                    # print(a)
                    # print('qqqqqqqq')
                    # print(b)
                    a[:, i] = a[:, i] + a[:, j]*b[:, i-j-1]
                a[:, i] = a[:, i]/i
                # print(a)
                # calculate p(:,i)

                for j in range(0, n):
                    p[j, i] = concat(i, d[j, :], a[j, :])
                    # print(m,i,j,tt)
                    # if m==1:
                        # print(i,j,m,p[j,i],d[j,:],a[j,:])

                if i == 1:
                    p[:, i] = p[:, i] - 1

                # calculate RHS of main equation
                for j in range(1, i+1):
                    # print('is neg', i-j-1)
                    c1[j-1, :] = b[:, i-j]

                c2 = np.dot(p[:, 1:i+1], c1[0:i, :])

                for j in range(0, n):
                    bigb[j, 0] = np.sum(b1[j, :] * c2[j, :])

                # calculate new b
                b2[:, :] = np.dot(biga, bigb)
                b[:, i] = b2[:, 0]


            m += 1
            # calculate new values of l and inverse bids, and lpl
            for i in range(0, big_J+1):

                l[:, m] += a[:, i] * ((-inc)**i)
                lpl[:, m] += b[:, i] * ((-inc)**i)
                bids[:, m-1] += p[:, i] * ((-inc)**i)

                # print(m,bids[0,m+1],(-inc)**i,i,p[0,i])

            # print(l[:,m],m,cdfres)
            check += np.where(l[:, m] - cdfres < 0, 1, 0)
            # print(l[:, m])
            # print(l[:, m+1])
            check += np.where(l[:, m] > 1, 1, 0)
            check += np.where(l[:, m] > l[:, m-1], 1, 0)
            # print(check)
            # print(l)
            # print(lpl)
            if np.sum(check) > 0:
                # print(m)
                # print(sum(np.where(l[:, m] - cdfres < 0, 1, 0)))
                # print(sum( np.where(l[:, m] > 1, 1, 0)))
                # print(sum(np.where(l[:, m] > l[:, m+1], 1, 0)))
                obj1 = 1000
                # print(m)
                m = nt
                # print("check exit")
            elif sum(check) == 0 and m > nt-2:
                obj1 += np.sqrt(np.sum(p[:, 0]**2))

        bad_form.bids = bids
        bad_form.lpl = lpl
        return obj1


    def best_response():
        vgrid = np.zeros((nt+1))
        mat1 = np.zeros((nt+1))
        mat0 = np.zeros((nt+1))
        kstar = np.zeros(n)
        slpl = np.zeros((nt+1))
        brr = np.zeros((2*n, nt+1))
        lpl = bad_form.lpl
        vgrid = np.linspace(lv, fires, nt+1)
        vgrid[0] = lv
        vgrid[-1] = fires

        h = 0
        for i in range(0, n):
            kstar = np.ones(n)
            kstar[i]-=1
            # print(kstar)
            slpl = np.zeros((nt+1))
            for j in range(0,n):
                slpl += kstar[j]*bad_form.lpl[j,:]
                # print(i,j,kstar[j],lpl[j,:])
            # print(kstar[i], t)
            for ns in range(1, nt):
                xx = vgrid[ns]
                mat0 = (1-(t-xx)*slpl)

                mat1 = mat0**2
                ml = np.argmin(mat1)

                brr[h, ns] = xx
                brr[h+1, ns] = t[ml]
            h+=2
        bad_form.brr = brr





    for i in range(ngrid-2, 0,-1):
        obj[i] = asym_precursion(t_grid[i])
        # print(t_grid[i])
        # print(obj[i])
        if obj[i] > obj[i+1]:
            tstar = t_grid[i+1]
            print("exiting b")
            break
    # print(i)
    tstar = t_grid[i+1]
    # print("t*", tstar)
    ginc = (uv-res)/(ngrid-1)
    xint = ginc
    dogrid = False
    # print(obj)
    fires = res
    # print(bad_form.bids.shape)


    result = minimize(asym_precursion, tstar, method='nelder-mead', options={'ftol': 1e-8, 'disp': True})
    print(result.x)
    tstar = result.x[0]
    t = np.linspace(tstar, res, nt+1)
    all_bids.append(bad_form.bids)
    all_tstars.append(t)
    # ta = np.array([t for i in range(0,n)])
    # best_response()

    # print(bad_form.bids[:, :])
    # plt.plot(bad_form.bids[:, :].T,ta.T)
    # plt.plot(bad_form.bids[:, :].T,ta.T)
    print(bad_form.bids[0,0],bad_form.bids[0,-1])
x = np.linspace(left, right, nt+1)
bz = np.zeros((step_count,nt+1))
for i in range(step_count):

    y_vals = np.interp(x, all_bids[i][0,:],all_tstars[i])
    bz[i,:] = np.where(x > all_bids[i][0,-1],x,y_vals)
    plt.plot(x, bz[i, :])

# final = np.mean(bz,axis=0)
# print(x)
# print(final)
# plt.plot(x, final)
plt.savefig('huehuehue_all.png')
# np.savetxt('final_all.csv', [x, final], fmt='%.5f',  delimiter=',')