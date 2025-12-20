import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from . import modelselect_alignment as alignment

def initialize_theta(d2: np.ndarray, d1: np.ndarray) -> Tuple[np.ndarray,str]:
	#### Initialize/Load/Find polynomial transforms for D2 into D1
	theta = alignment.alignment_guess_coefficients(d2,d1)
	log = f'\tGuessed Fourier shift: ({theta[0]:.4f}, {theta[3]:.4f})\n'
	return theta,log

def optimize_data(d2: np.ndarray, d1: np.ndarray, theta: np.ndarray, order: int, downscale: float, maxiter: int, miniter: int) -> Tuple[np.ndarray,str]:
	'''
	Moving d2 into d1. Optimize from theta as an initial guess
	order is polynomial order for transform. Int: [1,2,3,4,...]
	downscale is the factor to downscale by. Power of two. Float: [1.0,2.0,4.0,8.0]
	miniter,maxiter control the outloop for optimization. At least miniter, at most maxiter rounds completed.
	'''
		
	#### Downscale image
	orig_shape = d1.shape
	for i in range(int(np.log2(downscale))):
		d1 = alignment.downscale_img(d1)
		d2 = alignment.downscale_img(d2)
	log = f'\tDownscaled {int(np.log2(downscale))} times: {orig_shape} >> {d1.shape}\n'

	## Scale theta to the proper downscaled image space
	theta = alignment.upscale_theta(theta,1./downscale)

	### Move to the requested order
	while alignment.coefficients_order(theta) > order:
		theta = alignment.coefficients_decrease_order(theta)
	while alignment.coefficients_order(theta) < order:
		theta = alignment.coefficients_increase_order(theta)

	if alignment.coefficients_order(theta) < 1:
		raise Exception('Order too low')
	log += f'\tTarget polynomial order: {order}\n'
	
	## Optimize
	print('Optimization\n========================')
	iter = 0
	for iter in range(maxiter):
		theta,result = alignment.alignment_max_evidence_polynomial(d2, d1, theta, maxiter=10000, progressbar=True)
		sout = f'\tIteration: {iter}, Success:{result.success}, Distorted:{alignment.check_distorted(d2, theta)}, Score:{-result.fun}'
		log += sout+'\n'
		if result.success and iter >= miniter:
			break
		if iter >= maxiter - 1 and not result.success:
			raise Exception('Failed!!! Order:%d, Iteration:%d'%(alignment.coefficients_order(theta),iter))
	print('========================')

	theta = alignment.upscale_theta(theta, c=downscale)
	log += f'\tExcessively Distorted Image? {alignment.check_distorted(np.empty(orig_shape),theta)}\n'
	return theta,log

def render_images(d2: np.ndarray, d1: np.ndarray, theta: np.ndarray):	
	if theta is None:
		theta = alignment.coefficients_blank(1)

	imgrgb0 = alignment.nps2rgb(d1,d2)

	#### Output Images
	qg = alignment.rev_interpolate_polynomial(d1,*alignment.coefficients_split(theta))
	imgrgb1 = alignment.nps2rgb(qg,d2)

	qr = np.zeros_like(d2)
	ll = qr.shape[0]//10
	for i in range(qr.shape[0]//ll):
		qr[ll//2 + i*ll,:] = 1.
	ll = qr.shape[1]//10
	for j in range(qr.shape[1]//ll):
		qr[:,ll//2 + j*ll] = 1.

	qg = qr.copy()
	qg = alignment.rev_interpolate_polynomial(qg,*alignment.coefficients_split(theta))
	imgrgb2 = alignment.nps2rgb(qg,qr)

	## avoid warnings by clipping at 0 and 1
	imgrgb0[imgrgb0<0.] = 0
	imgrgb0[imgrgb0>1.] = 1.
	imgrgb1[imgrgb1<0.] = 0
	imgrgb1[imgrgb1>1.] = 1.
	imgrgb2[imgrgb2<0.] = 0
	imgrgb2[imgrgb2>1.] = 1.

	## make plot
	fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
	ax[0].imshow(imgrgb0, interpolation='nearest')
	ax[1].imshow(imgrgb1, interpolation='nearest')
	ax[2].imshow(imgrgb2, interpolation='nearest')
	return fig,ax
