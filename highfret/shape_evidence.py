import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import numpy as np
np.seterr(all='ignore')
from .jit_math import betainc, gammaln
import numba as nb

lnprior_slope = np.log(10000.) ## 10k max photons
lnprior_offset = np.log(1000.) ## offset is < 1000
lnprior_scale = np.log(np.log(1000.)-np.log(1.))

@nb.njit(cache=True)
def ln_evidence(x,y):
	### 2.2.2

	if y.size == 0:
		return -np.inf

	N = float(x.size)
	M = N/2.-1.
	
	Ex = np.mean(x)
	Ey = np.mean(y)
	Exx = np.mean(x*x)
	Eyy = np.mean(y*y)
	Exy = np.mean(x*y)
	vx = Exx-Ex*Ex
	vy = Eyy-Ey*Ey
	vxy = Exy-Ex*Ey

	## underflow protection
	if vx <= 0 or vy <= 0 or vxy == 0: ## vxy can be negative
		return -np.inf

	r = vxy/np.sqrt(vx*vy)
	r2 = r*r
	out = gammaln(M) -N/2.*np.log(N) -.5*np.log(vx) -np.log(2) - lnprior_slope - lnprior_offset - lnprior_scale -M*np.log(np.pi) -M*np.log(vy) -M*np.log(1.-r2) + np.log(1.+r/np.abs(r)*betainc(.5,M,r2))
	return out

@nb.njit(cache=True)
def null_ln_evidence(y):
	## 2.2.4

	if y.size == 0:
		return -np.inf

	N = float(y.size)
	M = N/2.-1.
	Ey = np.mean(y)
	Eyy = np.mean(y*y)
	vy = Eyy-Ey*Ey

	## underflow protection
	if vy <= 0:
		return -np.inf

	out = gammaln(M+.5) -N/2.*np.log(N) -lnprior_offset -lnprior_scale -(M+.5)*np.log(np.pi) -(M+.5)*np.log(vy)
	return out

# @nb.njit
def calc_lnevidences(templates,data):
	nt,ntx,nty = templates.shape
	nx,ny = data.shape

	ll = (ntx-1)//2
	ls = np.zeros((nt+1,nx,ny))

	for i in range(nx):
		for j in range(ny):
			min1 = max(0,i-ll)
			max1 = min(nx-1,i+ll+1)
			min2 = max(0,j-ll)
			max2 = min(ny-1,j+ll+1)
			dmin1 = min1 - (i-ll)
			dmax1 = (i+ll+1) - max1
			dmin2 = min2 - (j-ll)
			dmax2 = (j+ll+1) - max2
			for t in range(nt):
				ls[t,i,j] = ln_evidence(templates[t,dmin1:ntx-dmax1,dmin2:nty-dmax2],data[min1:max1,min2:max2])
			ls[-1,i,j] = null_ln_evidence(data[min1:max1,min2:max2])
	return ls