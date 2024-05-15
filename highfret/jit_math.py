import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import numba as nb
from math import log,fabs,lgamma,exp,gamma,pow,nan
from math import erf as erff

@nb.vectorize(cache=True)
def erf(x):
	return erff(x)

@nb.vectorize(cache=True)
def gammaln(x):
	return lgamma(x)

'''
/*
 * Cephes Math Library, Release 2.3:  March, 1995
 * Copyright 1984, 1995 by Stephen L. Moshier
 */

rewriting incbet from Cephes into numba jit-able function
CKT Feb 02, 2022

'''

### UNK(nown) mode from const.c
MAXGAM = 171.624376956302725
MACHEP =  1.38777878078144567553E-17 #2**-56
MAXLOG =  8.8029691931113054295988E1  #log(2**127)
MINLOG = -8.872283911167299960540E1 #log(2**-128)
MAXNUM =  1.701411834604692317316873e38 #2**127

big = 4.503599627370496e15
biginv = 2.22044604925031308085e-16

#
# /* Power series for incomplete beta integral.
#  * Use when b*x is small and x not too close to 1.  */
@nb.vectorize(["f8(f8,f8,f8)"],cache=True)
def pseries(a, b, x):
	# double s, t, u, v, n, t1, z, ai;

	ai = 1.0 / a
	u = (1.0 - b) * x
	v = u / (a + 1.0)
	t1 = v
	t = u
	n = 2.0
	s = 0.0
	z = MACHEP * ai
	while (fabs(v) > z):
		u = (n - b) * x / n
		t *= u
		v = t / (a + n)
		s += v
		n += 1.0
	s += t1
	s += ai

	u = a * log(x)
	if ((a + b) < MAXGAM and fabs(u) < MAXLOG):
		t = gamma(a+b)/gamma(a)/gamma(b)
		s = s * t * pow(x, a)
	else:
		t = lgamma(a+b)-lgamma(a)-lgamma(b) + u + log(s)
		if (t < MINLOG):
			s = 0.0
		else:
			s = exp(t)
	return s


# /* Continued fraction expansion #1
#  * for incomplete beta integral
#  */

@nb.vectorize(["f8(f8,f8,f8)"],cache=True)
def incbcf(a, b, x):
	# double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
	# double k1, k2, k3, k4, k5, k6, k7, k8;
	# double r, t, ans, thresh;
	# int n;

	k1 = a
	k2 = a + b
	k3 = a
	k4 = a + 1.0
	k5 = 1.0
	k6 = b - 1.0
	k7 = k4
	k8 = a + 2.0

	pkm2 = 0.0
	qkm2 = 1.0
	pkm1 = 1.0
	qkm1 = 1.0
	ans = 1.0
	r = 1.0
	n = 0
	thresh = 3.0 * MACHEP
	while n < 300:
		n += 1

		xk = -(x * k1 * k2) / (k3 * k4)
		pk = pkm1 + pkm2 * xk
		qk = qkm1 + qkm2 * xk
		pkm2 = pkm1
		pkm1 = pk
		qkm2 = qkm1
		qkm1 = qk

		xk = (x * k5 * k6) / (k7 * k8)
		pk = pkm1 + pkm2 * xk
		qk = qkm1 + qkm2 * xk
		pkm2 = pkm1
		pkm1 = pk
		qkm2 = qkm1
		qkm1 = qk

		if (qk != 0):
			r = pk / qk

		if (r != 0):
			t = fabs((ans - r) / r)
			ans = r
		else:
			t = 1.0

		if (t < thresh):
			n = 301

		k1 += 1.0
		k2 += 1.0
		k3 += 2.0
		k4 += 2.0
		k5 += 1.0
		k6 -= 1.0
		k7 += 2.0
		k8 += 2.0

		if ((fabs(qk) + fabs(pk)) > big):
			pkm2 *= biginv
			pkm1 *= biginv
			qkm2 *= biginv
			qkm1 *= biginv

		if ((fabs(qk) < biginv) or (fabs(pk) < biginv)):
			pkm2 *= big
			pkm1 *= big
			qkm2 *= big
			qkm1 *= big

	return ans


# /* Continued fraction expansion #2
#  * for incomplete beta integral
#  */

@nb.vectorize(["f8(f8,f8,f8)"],cache=True)
def incbd(a, b, x):
	# double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
	# double k1, k2, k3, k4, k5, k6, k7, k8;
	# double r, t, ans, z, thresh;
	# int n;

	k1 = a
	k2 = b - 1.0
	k3 = a
	k4 = a + 1.0
	k5 = 1.0
	k6 = a + b
	k7 = a + 1.0
	k8 = a + 2.0

	pkm2 = 0.0
	qkm2 = 1.0
	pkm1 = 1.0
	qkm1 = 1.0
	z = x / (1.0 - x)
	ans = 1.0
	r = 1.0
	n = 0
	thresh = 3.0 * MACHEP
	while n < 300:
		n += 1

		xk = -(z * k1 * k2) / (k3 * k4)
		pk = pkm1 + pkm2 * xk
		qk = qkm1 + qkm2 * xk
		pkm2 = pkm1
		pkm1 = pk
		qkm2 = qkm1
		qkm1 = qk

		xk = (z * k5 * k6) / (k7 * k8)
		pk = pkm1 + pkm2 * xk
		qk = qkm1 + qkm2 * xk
		pkm2 = pkm1
		pkm1 = pk
		qkm2 = qkm1
		qkm1 = qk

		if (qk != 0):
			r = pk / qk

		if (r != 0):
			t = fabs((ans - r) / r)
			ans = r
		else:
			t = 1.0

		if (t < thresh):
			n = 301

		k1 += 1.0
		k2 -= 1.0
		k3 += 2.0
		k4 += 2.0
		k5 += 1.0
		k6 += 1.0
		k7 += 2.0
		k8 += 2.0

		if ((fabs(qk) + fabs(pk)) > big):
			pkm2 *= biginv
			pkm1 *= biginv
			qkm2 *= biginv
			qkm1 *= biginv
		if ((fabs(qk) < biginv) or (fabs(pk) < biginv)):
			pkm2 *= big
			pkm1 *= big
			qkm2 *= big
			qkm1 *= big
	return ans

@nb.vectorize(["f8(f8,f8,f8)"],cache=True)
def betainc(aa, bb, xx):
	# double a, b, t, x, xc, w, y;
	# int flag;

	if (aa <= 0.0 or bb <= 0.0):
		return nan

	if ((xx <= 0.0) or (xx >= 1.0)):
		if (xx == 0.0):
			return 0.0
		if (xx == 1.0):
			return 1.0
		return nan

	flag = 0;
	if ((bb * xx) <= 1.0 and xx <= 0.95):
		t = pseries(aa, bb, xx)
		return t

	w = 1.0 - xx;

	## /* Reverse a and b if x is greater than the mean. */
	if (xx > (aa / (aa + bb))):
		flag = 1
		a = bb
		b = aa
		xc = xx
		x = w
	else:
		a = aa
		b = bb
		xc = w
		x = xx

	if (flag == 1 and (b * x) <= 1.0 and x <= 0.95):
		t = pseries(a, b, x)
		if (t <= MACHEP):
			t = 1.0 - MACHEP
		else:
			t = 1.0 - t
		return t

	## /* Choose expansion for better convergence. */
	y = x * (a + b - 2.0) - (a - 1.0)
	if (y < 0.0):
		w = incbcf(a, b, x)

	else:
		w = incbd(a, b, x) / xc

	## /* Multiply w by the factor
	##  * a	  b   _			 _	 _
	##  * x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

	y = a * log(x)
	t = b * log(xc)
	if ((a + b) < MAXGAM and fabs(y) < MAXLOG and fabs(t) < MAXLOG):
		t = pow(xc, b)
		t *= pow(x, a)
		t /= a
		t *= w
		t *= gamma(a+b)/gamma(a)/gamma(b)
		if (flag == 1):
			if (t <= MACHEP):
				t = 1.0 - MACHEP
			else:
				t = 1.0 - t
		return t

	## /* Resort to logarithms.  */
	y += t - lgamma(a) - lgamma(b) + lgamma(a+b)
	y += log(w / a)
	if (y < MINLOG):
		t = 0.0
	else:
		t = exp(y)

	## done
	if (flag == 1):
		if (t <= MACHEP):
			t = 1.0 - MACHEP
		else:
			t = 1.0 - t
	return t




if __name__ == '__main__':
	import time
	import numpy as np
	from scipy.special import betainc as _betainc

	def accuracy_test():
		## cephes
		betainc(.5,.5,.5) ## compile it....
		print('')
		print('RMSD(JIT v CEPHES),RMSD(CEPHES IEEE),t(SciPy),t(JIT)')
		for domain,trials,rmsd_ieee in zip([5,85,1000,10000,100000],[10000,250000,30000,250000,10000],[4.5e-16,1.7e-14,6.3e-13,7.1e-12,4.8e-11]):
			x = np.random.rand(trials)
			a,b = np.random.rand(2,trials)*domain
			t0 = time.time()
			y = betainc(a,b,x)
			# y = np.array([betainc(a[i],b[i],x[i]) for i in range(x.size)])
			t1 = time.time()
			_y = _betainc(a,b,x)
			t2 = time.time()
			print('%.2e %.2e, %4.4f %4.4f'%(np.sqrt(np.mean((y-_y)**2.)),rmsd_ieee,t2-t1,t1-t0))

	accuracy_test()
