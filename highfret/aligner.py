import os
import time
import numpy as np
import matplotlib.pyplot as plt

from . import prepare
from . import modelselect_alignment as alignment
order = alignment.coefficients_order


def prepare_data(fn_data,flag_what,flag_split,first=0,last=0):
	#### Prepare

	d = prepare.load(fn_data)
	end = d.shape[0] if last == 0 else last
	if first >= end:
		first = end-1
	d = d[first:end]
	print('Loaded %s'%(fn_data))
	print('Data Shape:',d.shape)

	if d.ndim == 3:
		## normalizations are mostly for visualizations!
		if flag_what == 'mean':
			img = d.mean(0)
			print('Using Average of %d frames'%(d.shape[0]))
		elif flag_what == 'acf':
			img = prepare.acf(d)
			print('Using ACF(t=1) image of %d frames'%(d.shape[0]))
		elif type(flag_what) is int:
			if flag_what >= d.shape[0]:
				raise Exception('No such frame. %s'%(str(d.shape)))
			img = d[flag_what].astype('double')
			print('Using frame %d'%(flag_what))
		else:
			raise Exception('do not know how to interpret preparation instruction')
	elif d.ndim == 2:
		img = d.astype('double')

	if flag_split == 'L/R':
		print('Split image L/R')
		dg,dr = prepare.split_lr(img)
	elif flag_split == 'T/B':
		print('Split image T/B')
		dg,dr = prepare.split_tb(img)

	dirs = prepare.get_out_dir(fn_data)
	dir_temp = dirs[1]
	dir_aligner = dirs[2]

	print('Prepared Shapes: %s, %s'%(str(dg.shape),str(dr.shape)))
	np.save(os.path.join(dir_temp,'align_dg.npy'),dg)
	np.save(os.path.join(dir_temp,'align_dr.npy'),dr)

	with open(os.path.join(dir_aligner,'details_preparation.txt'),'w') as f:
		out = "Aligner - %s\n=====================\n"%(time.ctime())
		out += '%s \n'%(fn_data)
		out += '%s \n=====================\n'%(str(d.shape))
		if flag_split == 'L/R':
			out += 'Split: Left/Right\n'
		else:
			out += 'Split: Top/Bottom\n'
		if d.ndim == 3:
			if flag_what == 'mean':
				out += 'Image Procesing: Temporal Mean\n'
			elif flag_what == 'acf':
				out += 'Image Procesing: Autocorrelation Function Image (tau = 1 frame)\n'
			else:
				out += 'Image Procesing: Frame %d\n'%(flag_what)
		f.write(out)
	print('Prepared data')

def initialize_theta(fn_data,flag_load,fn_theta,flag_fourier_guess):
	#### Initialize/Load/Find polynomial transforms for R into G
	if flag_load:
		if os.path.exists(fn_theta):
			theta = np.load(fn_theta)
			print('Loaded %s'%(fn_theta))
			print('Loaded Order: %d'%(order(theta)))
		else:
			raise Exception('File does not exist: %s'%(fn_theta))
	else:
		if flag_fourier_guess:
			dg,dr = get_prepared_data(fn_data)
			theta = alignment.alignment_guess_coefficients(dr,dg)
			# theta = alignment.upscale_theta(theta, c=flag_downscale)
			print('Guessed Fourier shift: (%.4f, %.4f)'%(theta[0],theta[3]))
		else:
			theta = alignment.coefficients_blank(1)
			print('Starting with blank coefficients')

	return theta

def get_prepared_data(fn_data):
	#### Load prepared image
	dirs = prepare.get_out_dir(fn_data)
	dir_temp = dirs[1]
	dir_aligner = dirs[2]

	if not os.path.exists(os.path.join(dir_temp,'align_dg.npy')) or not os.path.exists(os.path.join(dir_temp,'align_dr.npy')):
		raise Exception('Please run prepare_data first')
		
	dg = np.load(os.path.join(dir_temp,'align_dg.npy'))
	dr = np.load(os.path.join(dir_temp,'align_dr.npy'))

	return dg,dr

def optimize_data(fn_data,theta,flag_downscale,flag_order,flag_optimize,flag_maxiter,flag_miniter):
	dg,dr = get_prepared_data(fn_data)
	
	#### Downscale image
	orig_shape = dg.shape
	for i in range(int(np.log2(flag_downscale))):
		dg = alignment.downscale_img(dg)
		dr = alignment.downscale_img(dr)
	print('Downscaled %d times: %s >> %s'%(int(np.log2(flag_downscale)),str(orig_shape),str(dg.shape)))

	## Scale theta to the proper downscaled image space
	theta = alignment.upscale_theta(theta,1./flag_downscale)

	### Move to the requested order
	while order(theta) > flag_order:
		theta = alignment.coefficients_decrease_order(theta)
	while order(theta) < flag_order:
		theta = alignment.coefficients_increase_order(theta)

	if order(theta) < 1:
		raise Exception('Order too low')
	print('Target polynomial order:',flag_order)
	
	if flag_optimize:
		print('Optimization\n========================')
		iter = 0
		for iter in range(flag_maxiter):
			theta,result = alignment.alignment_max_evidence_polynomial(dr,dg,theta,maxiter=10000,progressbar=True)
			print(order(theta),result.success,alignment.check_distorted(dr,theta),-result.fun)
			if result.success and iter >= flag_miniter:
				break
			if iter >= flag_maxiter - 1 and not result.success:
				raise Exception('Failed!!! Order:%d, Iteration:%d'%(order(theta),iter))
		print('========================')

	theta = alignment.upscale_theta(theta, c=flag_downscale)
	print('Excessively Distorted Image?',alignment.check_distorted(np.empty(orig_shape),theta))

	return theta

def render_images(fn_data,theta=None):
	dg,dr = get_prepared_data(fn_data)
	
	if theta is None:
		theta = alignment.coefficients_blank(1)

	imgrgb0 = alignment.nps2rgb(dg,dr)

	#### Output Images
	qg = alignment.rev_interpolate_polynomial(dg,*alignment.coefficients_split(theta))
	imgrgb1 = alignment.nps2rgb(qg,dr)

	qr = np.zeros_like(dr)
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
	ax[0].imshow(imgrgb0,interpolation='nearest')
	ax[1].imshow(imgrgb1,interpolation='nearest')
	ax[2].imshow(imgrgb2,interpolation='nearest')
	return fig,ax
