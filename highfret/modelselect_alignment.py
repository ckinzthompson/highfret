import numpy as np
np.seterr(all='ignore')
import numba as nb

from scipy.optimize import minimize
from .shape_evidence import ln_evidence
from tqdm import tqdm


@nb.njit
def polynomial_get_order(a):
	''' figure out polynomial order from list of coefficients '''
	ncoef = a.size
	Kmax = 20
	sk = 0
	for K in range(Kmax):
		sk = sk + K+1
		if sk == ncoef:
			break
	if K == Kmax - 1:
		raise Exception('Wrong number of coefficients')
	return K

@nb.njit
def polynomial_transform(x1,y1,a,b,K):
	'''
	transforms q(x,y)->w(x',y') where:
		x' = sum_{j=0}^K sum_{i=0}^j \alpha_{ji} * x^(j-i) * y^i
		y' = sum_{j=0}^K sum_{i=0}^j \beta_{ji}  * x^(j-i) * y^i
	alpha and beta are 1d arrays listed in order of the sum
	(i.e. a00,a10,a11,a20,a21,a22,a30,a31,a32,a33...etc)
	'''
	## initialize for speed
	x2 = 0.
	x3 = 0.
	x4 = 0.
	x5 = 0.
	x6 = 0.
	y2 = 0.
	y3 = 0.
	y4 = 0.
	y5 = 0.
	y6 = 0.

	## loops are much slower than hard coding for some reason...
	if K >= 0:
		x = a[0]
		y = b[0]
	if K >= 1:
		x += a[1]*x1 + a[2]*y1
		y += b[1]*x1 + b[2]*y1
	if K >= 2:
		x2 = x1*x1
		y2 = y1*y1
		x += a[3]*x2 + a[4]*x1*y1 + a[5]*y2
		y += b[3]*x2 + b[4]*x1*y1 + b[5]*y2
	if K >= 3:
		x3 = x2*x1
		y3 = y2*y1
		x += a[6]*x3 + a[7]*x2*y1 + a[8]*x1*y2 + a[9]*y3
		y += b[6]*x3 + b[7]*x2*y1 + b[8]*x1*y2 + b[9]*y3
	if K >= 4:
		x4 = x3*x1
		y4 = y3*y1
		x +=  a[10]*x4 + a[11]*x3*y1 + a[12]*x2*y2 + a[13]*x1*y3 + a[14]*y4
		y +=  b[10]*x4 + b[11]*x3*y1 + b[12]*x2*y2 + b[13]*x1*y3 + b[14]*y4
	if K >= 5:
		x5 = x4*x1
		y5 = y4*y1
		x +=  a[15]*x5 + a[16]*x4*y1 + a[17]*x3*y2 + a[18]*x2*y3 + a[19]*x1*y4 + a[20]*y5
		y +=  b[15]*x5 + b[16]*x4*y1 + b[17]*x3*y2 + b[18]*x2*y3 + b[19]*x1*y4 + b[20]*y5
	if K >= 6:
		x6 = x5*x1
		y6 = y5*y1
		x +=  a[21]*x6 + a[22]*x5*y1 + a[23]*x4*y2 + a[24]*x3*y3 + a[25]*x2*y4 + a[26]*x1*y5 + a[27]*y6
		y +=  b[21]*x6 + b[22]*x5*y1 + b[23]*x4*y2 + b[24]*x3*y3 + b[25]*x2*y4 + b[26]*x1*y5 + b[27]*y6
	if K >= 7:
		x7 = x6*x1
		y7 = y6*y1
		x +=  a[27]*x7 + a[28]*x6*y1 + a[29]*x5*y2 + a[30]*x4*y3 + a[31]*x3*y4 + a[32]*x2*y5 + a[33]*x1*y6 + a[34]*y7
		y +=  b[27]*x7 + b[28]*x6*y1 + b[29]*x5*y2 + b[30]*x4*y3 + b[31]*x3*y4 + b[32]*x2*y5 + b[33]*x1*y6 + a[34]*y7
	if K >= 8:
		x8 = x7*x1
		y8 = y7*y1
		x +=  a[35]*x8 + a[36]*x7*y1 + a[37]*x6*y2 + a[38]*x5*y3 + a[39]*x4*y4 + a[40]*x3*y5 + a[41]*x2*y6 + a[42]*y8
		y +=  b[35]*x8 + b[36]*x7*y1 + b[37]*x6*y2 + b[38]*x5*y3 + b[39]*x4*y4 + b[40]*x3*y5 + b[41]*x2*y6 + a[42]*y8
	return x,y

@nb.njit
def interpolate_polynomial(q,a,b):
	'''
	transforms q(x,y)->w(x',y') where:
		x' = sum_{j=0}^K sum_{i=0}^j \alpha_{ji} * x^(j-i) * y^i
		y' = sum_{j=0}^K sum_{i=0}^j \beta_{ji}  * x^(j-i) * y^i
	alpha and beta are 1d arrays listed in order of the sum
	(i.e. a00,a10,a11,a20,a21,a22,a30,a31,a32,a33...etc)
	Once the pixel containing (x',y') is found, a bilinear interpolation on the nearest neighbor pixels is used to get the corresponding value... not ideal, but good enough
	'''

	## initialize everything for speed
	out = np.zeros_like(q) + np.nan
	ii = 0
	jj = 0
	x = 0.
	y = 0.
	x1 = 0.
	x2 = 0.
	x3 = 0.
	x4 = 0.
	x5 = 0.
	x6 = 0.
	y1 = 0.
	y2 = 0.
	y3 = 0.
	y4 = 0.
	y5 = 0.
	y6 = 0.

	K = polynomial_get_order(a)
	if a.size != b.size:
		raise Exception('Coefficients mismatched')
	if K > 8:
		raise Exception('Not Implemented; too high degree polynomial')

	## make the new image
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			x1 = float(i)
			y1 = float(j)

			## I think it's slower to call another function here rather than embed
			if K >= 0:
				x = a[0]
				y = b[0]
			if K >= 1:
				x += a[1]*x1 + a[2]*y1
				y += b[1]*x1 + b[2]*y1
			if K >= 2:
				x2 = x1*x1
				y2 = y1*y1
				x += a[3]*x2 + a[4]*x1*y1 + a[5]*y2
				y += b[3]*x2 + b[4]*x1*y1 + b[5]*y2
			if K >= 3:
				x3 = x2*x1
				y3 = y2*y1
				x += a[6]*x3 + a[7]*x2*y1 + a[8]*x1*y2 + a[9]*y3
				y += b[6]*x3 + b[7]*x2*y1 + b[8]*x1*y2 + b[9]*y3
			if K >= 4:
				x4 = x3*x1
				y4 = y3*y1
				x +=  a[10]*x4 + a[11]*x3*y1 + a[12]*x2*y2 + a[13]*x1*y3 + a[14]*y4
				y +=  b[10]*x4 + b[11]*x3*y1 + b[12]*x2*y2 + b[13]*x1*y3 + b[14]*y4
			if K >= 5:
				x5 = x4*x1
				y5 = y4*y1
				x +=  a[15]*x5 + a[16]*x4*y1 + a[17]*x3*y2 + a[18]*x2*y3 + a[19]*x1*y4 + a[20]*y5
				y +=  b[15]*x5 + b[16]*x4*y1 + b[17]*x3*y2 + b[18]*x2*y3 + b[19]*x1*y4 + b[20]*y5
			if K >= 6:
				x6 = x5*x1
				y6 = y5*y1
				x +=  a[21]*x6 + a[22]*x5*y1 + a[23]*x4*y2 + a[24]*x3*y3 + a[25]*x2*y4 + a[26]*x1*y5 + a[27]*y6
				y +=  b[21]*x6 + b[22]*x5*y1 + b[23]*x4*y2 + b[24]*x3*y3 + b[25]*x2*y4 + b[26]*x1*y5 + b[27]*y6
			if K >= 7:
				x7 = x6*x1
				y7 = y6*y1
				x +=  a[27]*x7 + a[28]*x6*y1 + a[29]*x5*y2 + a[30]*x4*y3 + a[31]*x3*y4 + a[32]*x2*y5 + a[33]*x1*y6 + a[34]*y7
				y +=  b[27]*x7 + b[28]*x6*y1 + b[29]*x5*y2 + b[30]*x4*y3 + b[31]*x3*y4 + b[32]*x2*y5 + b[33]*x1*y6 + a[34]*y7
			if K >= 8:
				x8 = x7*x1
				y8 = y7*y1
				x +=  a[35]*x8 + a[36]*x7*y1 + a[37]*x6*y2 + a[38]*x5*y3 + a[39]*x4*y4 + a[40]*x3*y5 + a[41]*x2*y6 + a[42]*y8
				y +=  b[35]*x8 + b[36]*x7*y1 + b[37]*x6*y2 + b[38]*x5*y3 + b[39]*x4*y4 + b[40]*x3*y5 + b[41]*x2*y6 + a[42]*y8

			# if x >= 0 and x+1 < q.shape[0] and y >= 0 and y+1 < q.shape[1]:
			# 	## interpolate - bilinear but with blank edges
			# 	ii = int(x//1)
			# 	jj = int(y//1)
			# 	x1 = float(ii)
			# 	y1 = float(jj)
			# 	x2 = x1+1.
			# 	y2 = y1+1.
			# 	out[i,j] = (x2-x)*q[ii,jj]*(y2-y) + (x2-x)*q[ii,jj+1]*(y-y1) + (x-x1)*q[ii+1,jj]*(y2-y) + (x-x1)*q[ii+1,jj+1]*(y-y1)
			## interpolate - bilinear but with sticky edges
			ii = int(x//1)
			if ii < 0: ii = 0
			if ii >= q.shape[0] - 1:
				ii = q.shape[0] - 2
			jj = int(y//1)
			if jj < 0: jj = 0
			if jj >= q.shape[1] - 1:
				jj = q.shape[1] - 2
			x1 = float(ii)
			y1 = float(jj)
			x2 = x1+1.
			y2 = y1+1.
			out[i,j] = (x2-x)*q[ii,jj]*(y2-y) + (x2-x)*q[ii,jj+1]*(y-y1) + (x-x1)*q[ii+1,jj]*(y2-y) + (x-x1)*q[ii+1,jj+1]*(y-y1)
	return out

@nb.njit
def interpolate_linearshift(q,dx,dy):
	a = np.array((dx,1.,0.))
	b = np.array((dy,0.,1.))
	return interpolate_polynomial(q,a,b)


def check_distorted(q,theta,factor=.5):
	## if True, bad image;
	## test that the image isn't crazy distorted
	a,b = coefficients_split(theta)
	K = polynomial_get_order(a)
	return _check_distorted(a,b,K,q,theta,factor)

@nb.njit
def _check_distorted(a,b,K,q,theta,factor):
	div = 4
	ll1 = q.shape[0]//div
	ll2 = q.shape[1]//div
	if q.shape[0] > q.shape[1]:
		r0 = q.shape[0]
	else:
		r0 = q.shape[1]
	for i in range(div):
		for j in range(div):
			x = float(ll1//2+ll1*i)
			y = float(ll2//2+ll2*j)
			xp,yp = polynomial_transform(x,y,a,b,K)
			r = np.sqrt((xp-x)**2. + (yp-y)**2.)
			if r > r0*factor:
				return True
	return False

def _fit_wrapper_poly(theta,d1,d2):
	a = theta[:theta.size//2]
	b = theta[theta.size//2:]
	if check_distorted(d1,theta):
		# raise Exception('Failure')
		return np.inf
	m2 = interpolate_polynomial(d1,a,b) ## approximate d2 using a transform of d1
	keep = np.isfinite(m2)
	# if keep.sum()/float(keep.size)<0.5: ## significant loss of overlap
	# 	return np.inf
	m2[np.bitwise_not(keep)] = d2.mean()
	# out = -ln_evidence(m2[keep],d2[keep]) ## see if the transformed d1 is a good model for d2
	# if np.isnan(out):
	# 	return np.inf
	# return out/float(keep.sum()) ## use avg/pixel b/c the nans change the data set so it's hard to compare
	out = -ln_evidence(m2,d2) ## see if the transformed d1 is a good model for d2
	return out/float(d2.size) ## avg for easily comparable numbers

def coefficients_combine(a,b):
	return np.concatenate((a,b))

def coefficients_blank(K=1):
	sk = np.sum(np.arange(K+1)+1) ## number of coefficients
	a = np.zeros(sk)
	b = np.zeros(sk)
	a[1] = 1.
	b[2] = 1.
	return coefficients_combine(a,b)

def coefficients_split(theta):
	alpha = theta[:theta.size//2]
	beta = theta[theta.size//2:]
	return alpha,beta

def coefficients_increase_order(c):
	a,b = coefficients_split(c)
	K = polynomial_get_order(a)
	aa,bb = coefficients_split(coefficients_blank(K+1))
	aa[:a.size] = a.copy()
	bb[:b.size] = b.copy()
	theta = coefficients_combine(aa,bb)
	return theta

def coefficients_decrease_order(c):
	a,b = coefficients_split(c)
	K = polynomial_get_order(a)
	if K > 1:
		aa,bb = coefficients_split(coefficients_blank(K-1))
		aa = a.copy()[:aa.size]
		bb = b.copy()[:bb.size]
		theta = coefficients_combine(aa,bb)
		return theta
	return c

def coefficients_order(theta):
	a,b = coefficients_split(theta)
	K = polynomial_get_order(a)
	return K

def alignment_upscaled_fft_phase(d1,d2):
	'''
	Find the linear shift in an image in phase space - upscales to super-resolve shift - d1 into d2

	input:
		* d1 - image 1 (Lx,Ly)
		* d2 - image 2 (Lx,Ly)
	output:
		* polynomial coefficients (K=1) with linear shift
	'''
	from skimage import transform
	from skimage.transform import SimilarityTransform,warp
	from scipy.interpolate import RectBivariateSpline

	## Calculate Cross-correlation
	f1 = np.fft.fft2(d1-d1.mean())
	f2 = np.fft.fft2(d2-d2.mean())
	f = np.conjugate(f1)*f2 # / (np.abs(f1)*np.abs(f2))
	d = np.fft.ifft2(f).real

	## Find maximum in c.c.
	s1,s2 = np.nonzero(d==d.max())

	#### Interpolate for finer resolution
	## cut out region
	l = 5.
	xmin = int(np.max((0.,s1[0]-l)))
	xmax = int(np.min((d.shape[0],s1[0]+l)))
	ymin = int(np.max((0.,s2[0]-l)))
	ymax = int(np.min((d.shape[1],s2[0]+l)))
	dd = d[xmin:xmax,ymin:ymax]

	## calculate interpolation
	x = np.arange(xmin,xmax)
	y = np.arange(ymin,ymax)
	interp = RectBivariateSpline(x,y,dd)

	## interpolate new grid
	x = np.linspace(xmin,xmax,(xmax-xmin)*10)
	y = np.linspace(ymin,ymax,(ymax-ymin)*10)
	di = interp(x,y)
	# return di

	## get maximum in interpolated c.c. for sub-pixel resolution
	xy = np.nonzero(di==di.max())

	## get interpolated shifts
	s1 = x[xy[0][0]]
	s2 = y[xy[1][0]]
	if s1 > d.shape[0]/2:
		s1 -= d.shape[0]
	if s2 > d.shape[1]/2:
		s2 -= d.shape[1]

	## calculate warped img
	tform = SimilarityTransform(translation=(-s2,-s1))
	transformed1 = warp(d1, tform)

	theta = coefficients_blank(1)
	a,b = coefficients_split(theta)
	a[0] = -s1
	b[0] = -s2
	return coefficients_combine(a,b)

def alignment_guess_coefficients(d1,d2,K=1,guess_linear=True):
	'''
	Convience function to generate coefficient arrays - d1 into d2
	input:
		* d1 - image 1 (Lx,Ly)
		* d2 - image 2 (Lx,Ly)
		* K - integer order of polynomial
		* guess_linear - flag: if True, will populate coefficients with linear shift guess
	output:
		* polynomial coefficients (order K)
	'''

	theta = coefficients_blank(K)
	a,b = coefficients_split(theta)

	if guess_linear:
		## get linear shift from upscaled FT
		coef_fft = alignment_upscaled_fft_phase(d1,d2)

		## Make grid +/-1, 20 by 20 points = .05 resolution
		ca,cb = coefficients_split(coef_fft)
		# xs = np.linspace(ca[0]-1.,ca[0]+1.,20)
		# ys = np.linspace(cb[0]-1.,cb[0]+1.,20)
		#
		# # get MAP solution on local grid
		# coef_grid = alignment_grid_evidence_linearshift(d1,d2,xs,ys)
		# ca,cb = coefficients_split(coef_grid)
		a[0] = ca[0]
		b[0] = cb[0]

	return coefficients_combine(a,b)

def alignment_max_evidence_polynomial(d1,d2,guess,maxiter=5000,progressbar=True,callback=None):
	'''
	Find the max evidence (/MAP) solution for the polynomial (order of guess) that overlaps d1 into d2
	uses Nelder-Mead
	if runs over maxiter, the result will have be result.success=False, but you collect the final point at result.x, and put it back into this function as a guess to start again.

	input:
		* d1 - image 1 (Lx,Ly)
		* d2 - image 2 (Lx,Ly)
		* guess - polynomial coefficient array that will be the starting point for the minimzer
		* maxiter - maximum iterations in the solver
		* progressbar - flag if true, displays TQDM progressbar for each iteration in maxiter
	output:
		* theta - polynomial coefficients (order K from input guess)
		* result - the scipy.optimize minimzer result of searching for the max evidence polynomial
	'''
	if progressbar:
		from tqdm import tqdm
		progress = tqdm(total=maxiter)
		result = minimize(_fit_wrapper_poly,x0=guess,args=(d1,d2),method='Nelder-Mead',options={'maxiter':maxiter},callback=lambda x:progress.update(1))
	else:
		result = minimize(_fit_wrapper_poly,x0=guess,args=(d1,d2),method='Nelder-Mead',options={'maxiter':maxiter},callback=callback)

	theta = result.x
	return theta, result


def nps2rgb(g,r):
	imgrgb = np.zeros((g.shape[0],g.shape[1],3))
	imgrgb[:,:,1] = g/np.percentile(g[5:-5,5:-5],99)
	imgrgb[:,:,0] = r/np.percentile(r[5:-5,5:-5],99)
	return imgrgb

def downscale_img(x):
	out = np.zeros((x.shape[0]//2,x.shape[1]//2))
	out += x[0:x.shape[0]-x.shape[0]%2:2,0:x.shape[1]-x.shape[1]%2:2]
	out += x[1:x.shape[0]-x.shape[0]%2:2,0:x.shape[1]-x.shape[1]%2:2]
	out += x[0:x.shape[0]-x.shape[0]%2:2,1:x.shape[1]-x.shape[1]%2:2]
	out += x[1:x.shape[0]-x.shape[0]%2:2,1:x.shape[1]-x.shape[1]%2:2]
	return out

def upscale_theta(theta,c=2.):
	'''
	so this works bc you can calculate the coefficients for a random linear scaling to x (think unit change)
	basically set up xhat = c*x
	x'=sum sum a_ji x^(j-i) y^i
	xhat' = sum sum b_ji xhat^(j-i) yhat^i = cx' = c sum sum a_ji x^(j-1)y^i
	match terms and find out
	
	a00 = 1/c b00
	a10 = b10, a11 = b11
	a20 = c b20, a21 = c b21, a22 = c b22
	a30 = c^2 b30, ....
	'''
	a,b = coefficients_split(theta)
	K = polynomial_get_order(a)
	for j in range(K+1):
		jj = j+1
		## this indexing uses that Gauss trick of 1 + 99, 2+98,...
		a[int(j/2*(j+1)):int(jj/2*(jj+1))] *= c**(-j+1)
		b[int(j/2*(j+1)):int(jj/2*(jj+1))] *= c**(-j+1)
	theta = coefficients_combine(a,b)
	return theta

