import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt

from . import prepare
from . import modelselect_alignment as alignment
order = alignment.coefficients_order


def get_out_dir(fn_data):
	filename = re.sub(r'\s+', '', fn_data)
	if os.path.exists(filename):

		## pull out file name from path
		fn_path,fn_data = os.path.split(filename)

		## pull off extension
		if fn_data.endswith('.ome.tif'):
			fn_base = fn_data[:-8]
		else:
			fn_base = os.path.splitext(fn_data)[0]

		fn_out_dir = os.path.join(fn_path,'aligner_results_%s'%(fn_base))
		return fn_out_dir
	else:
		raise Exception('File does not exist')


def prepare_data(fn_data,flag_what,flag_split):
	#### Prepare
	d = prepare.load(fn_data)
	print('Loaded %s'%(fn_data))
	print('Data Shape:',d.shape)

	if d.ndim == 3:
		## normalizations are mostly for visualizations!
		if flag_what == 'mean':
			# from .fast_median import med_perreault_boundary
			# img = np.zeros((d.shape[1],d.shape[2]))
			# for i in range(d.shape[0]):
			# 	img += med_perreault_boundary(d[i]//4,9).astype('float') #- med_perreault_boundary(d[i],10).astype('float')
			# 	break
			# # img /= float(d.shape[0])
			img = d.mean(0)
			# img = d.mean(0) - d.mean()
			# img /= np.sqrt(d.var(0)+d.var())
			print('Using Average of %d frames'%(d.shape[0]))
		elif flag_what == 'acf':
			img = prepare.acf1(d)
			print('Using ACF(t=1) image of %d frames'%(d.shape[0]))
		elif type(flag_what) is int:
			if flag_what >= d.shape[0]:
				raise Exception('No such frame. %s'%(str(d.shape)))
			img = d[flag_what].astype('double')
			img  -= img.mean()
			img /= img.std()
			print('Using frame %d'%(flag_what))
		else:
			raise Exception('do not know how to interpret preparation instruction')
	elif d.ndim == 2:
		img = d - d.mean()
		img /= img.std()

	if flag_split == 'L/R':
		print('Split image L/R')
		d1,d2 = prepare.split_lr(img)
	elif flag_split == 'T/B':
		print('Split image T/B')
		d1,d2 = prepare.split_tb(img)

	out_dir = get_out_dir(fn_data)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	print('Prepared Shapes: %s, %s'%(str(d1.shape),str(d2.shape)))
	np.save(os.path.join(out_dir,'prep_temp_d1.npy'),d1)
	np.save(os.path.join(out_dir,'prep_temp_d2.npy'),d2)

	with open(os.path.join(out_dir,'prep_details.txt'),'w') as f:
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
	#### Initialize/Load/Find polynomial transforms
	if flag_load:
		if os.path.exists(fn_theta):
			theta = np.load(fn_theta)
			print('Loaded %s'%(fn_theta))
			print('Loaded Order: %d'%(order(theta)))
		else:
			raise Exception('File does not exist: %s'%(fn_theta))
	else:
		if flag_fourier_guess:
			d1,d2 = get_prepared_data(fn_data)
			theta = alignment.alignment_guess_coefficients(d2,d1)
			# theta = alignment.upscale_theta(theta, c=flag_downscale)
			print('Guessed Fourier shift: (%.4f, %.4f)'%(theta[0],theta[3]))
		else:
			theta = alignment.coefficients_blank(1)
			print('Starting with blank coefficients')

	return theta

def get_prepared_data(fn_data):
	#### Load prepared image
	out_dir = get_out_dir(fn_data)

	if not os.path.exists(os.path.join(out_dir,'prep_temp_d1.npy')) or not os.path.exists(os.path.join(out_dir,'prep_temp_d2.npy')):
		raise Exception('Please run prepare_data first')
		
	d1 = np.load(os.path.join(out_dir,'prep_temp_d1.npy'))
	d2 = np.load(os.path.join(out_dir,'prep_temp_d2.npy'))

	return d1,d2

def optimize_data(fn_data,theta,flag_downscale,flag_order,flag_optimize,flag_maxiter,flag_miniter):
	d1,d2 = get_prepared_data(fn_data)
	
	#### Downscale image
	orig_shape = d1.shape
	for i in range(int(np.log2(flag_downscale))):
		d1 = alignment.downscale_img(d1)
		d2 = alignment.downscale_img(d2)
	print('Downscaled %d times: %s >> %s'%(int(np.log2(flag_downscale)),str(orig_shape),str(d1.shape)))

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
			theta,result = alignment.alignment_max_evidence_polynomial(d2,d1,theta,maxiter=10000,progressbar=True)
			print(order(theta),result.success,alignment.check_distorted(d2,theta),-result.fun)
			if result.success and iter >= flag_miniter:
				break
			if iter >= flag_maxiter - 1 and not result.success:
				raise Exception('Failed!!! Order:%d, Iteration:%d'%(order(theta),iter))
		print('========================')

	theta = alignment.upscale_theta(theta, c=flag_downscale)
	print('Excessively Distorted Image?',alignment.check_distorted(np.empty(orig_shape),theta))

	return theta

def render_images(fn_data,theta=None):
	d1,d2 = get_prepared_data(fn_data)
	
	if theta is None:
		theta = alignment.coefficients_blank(1)

	imgrgb0 = alignment.nps2rgb(d1,d2)

	#### Output Images
	q2 = alignment.interpolate_polynomial(d2,*alignment.coefficients_split(theta))
	imgrgb1 = alignment.nps2rgb(d1,q2)

	q1 = np.zeros_like(q2)
	ll = q1.shape[0]//10
	for i in range(q1.shape[0]//ll):
		q1[ll//2 + i*ll,:] = 1.
	ll = q1.shape[1]//10
	for j in range(q1.shape[1]//ll):
		q1[:,ll//2 + j*ll] = 1.

	q2 = q1.copy()
	q2 = alignment.interpolate_polynomial(q2,*alignment.coefficients_split(theta))
	imgrgb2 = alignment.nps2rgb(q1,q2)

	## avoid warnings by clipping at 0 and 1
	imgrgb0[imgrgb0<0.] = 0
	imgrgb0[imgrgb0>1.] = 1.
	imgrgb1[imgrgb1<0.] = 0
	imgrgb1[imgrgb1>1.] = 1.
	imgrgb2[imgrgb2<0.] = 0
	imgrgb2[imgrgb2>1.] = 1.

	## make plot
	fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
	ax[0].imshow(imgrgb0)#,origin='lower',interpolation='nearest')
	ax[1].imshow(imgrgb1)#,origin='lower',interpolation='nearest')
	ax[2].imshow(imgrgb2)#,origin='lower',interpolation='nearest')
	return fig,ax



if __name__ == '__main__':
	# fn_data = '/Users/colin/Desktop/projects/test_data/tirf/5.tif'
	fn_data = '/Users/colin/Desktop/projects/test_data/tirf/vc436HS_1_MMStack_Pos0.ome.tif'
	fn_data = '/Users/colin/Downloads/20240505_align_1_MMStack_Default.ome.tif'
	fn_theta = 'theta.npy'

	flag_what = 'mean'
	flag_split = 'lr'
	flag_downscale = 1.
	flag_order = 4
	flag_maxiter = 10
	flag_miniter = 5
	flag_fourier_guess = True
	flag_optimize = True
	flag_load = True
	flag_save = True

	if flag_maxiter < flag_miniter:
		flag_maxiter = flag_miniter

	prepare_data(fn_data,flag_what,flag_split)
	theta = initialize_theta(fn_data,flag_load,fn_theta,flag_fourier_guess)
	theta = optimize_data(fn_data,theta,flag_downscale,flag_order,flag_optimize,flag_maxiter,flag_miniter)
	if flag_save:
		np.save('theta.npy',theta)
	fig,ax = render_images(fn_data,theta)
	plt.show()