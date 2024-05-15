import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import os
import re
import h5py
import numpy as np
import numba as nb
from tqdm import tqdm
import matplotlib.pyplot as plt



from . import prepare,minmax,alignment,spotfinder

from scipy.special import erf

## numba doesn't really speed this up.....
def ml_psf(l,movie,sigma,xyi,maxiters=1000):
	try:
		xmin = int(max(0,xyi[0]-l))
		xmax = int(min(movie.shape[1]-1,xyi[0]+l) + 1)
		ymin = int(max(0,xyi[1]-l))
		ymax = int(min(movie.shape[2]-1,xyi[1]+l) + 1)

		gx,gy = np.mgrid[xmin:xmax,ymin:ymax]
		gx = gx.astype('f').flatten()
		gy = gy.astype('f').flatten()
		m = movie[:,xmin:xmax,ymin:ymax].astype('f').reshape((movie.shape[0],gx.size)).copy()

		## Find COM
		# xyi = com(m.sum(0)) + xyi - l

		dex = .5 * (erf((xyi[0]-gx+.5)/(np.sqrt(2.*sigma**2.))) - erf((xyi[0]-gx -.5)/(np.sqrt(2.*sigma**2.))))
		dey = .5 * (erf((xyi[1]-gy+.5)/(np.sqrt(2.*sigma**2.))) - erf((xyi[1]-gy -.5)/(np.sqrt(2.*sigma**2.))))
		psi = dex*dey
		psi /= psi.sum()

		b = np.min(m,axis=1)
		n = np.max(m,axis=1)-b

		n0 = n.sum()
		psum = np.sum(psi**2.)

		npm = np.median
		# npm = np.mean

		for iter in range(maxiters):
			b = npm(m - n[:,None]*psi[None,:],axis=(1)) ## median is much more stable
			n = np.sum((m - b[:,None])*psi[None,:],axis=(1))/psum
			n1 = n.sum()

			if np.abs((n1-n0)/n0) < 1e-5:
				break
			else:
				n0 = n1
	except:
		n = np.zeros(movie.shape[0])
		b = np.zeros(movie.shape[0])

	return n,b

def calculate_sigma(pixel_real,mag,bin,lambda_nm,NA,motion):
	pixel_xy = pixel_real/mag*bin
	sigma_xy = 0.21*lambda_nm/NA
	sigma_xy = np.sqrt(motion**2.+sigma_xy**2.)
	sigma = sigma_xy/pixel_xy
	return sigma

def load_spots(fn_data):
	print('Loading Spots')
	out_dir = spotfinder.get_out_dir(fn_data)
	prefix = os.path.split(out_dir)[1][19:]

	fn_spots_g = os.path.join(out_dir,'%s_g_spots.npy'%(prefix))
	fn_spots_r = os.path.join(out_dir,'%s_r_spots.npy'%(prefix))
	try:
		spots_g = np.load(fn_spots_g)
	except:
		raise Exception('Cannot load green spots: %s'%(fn_spots_g))
	try:
		spots_r = np.load(fn_spots_r)
	except:
		raise Exception('Cannot load red spots: %s'%(fn_spots_r))

	return spots_g,spots_r

def prepare_data(fn_data,fn_align,fn_cal=None,flag_split='L/R'):
	theta = np.load(fn_align)
	data = prepare.load(fn_data)
	if fn_cal is None:
		calibration = np.zeros((3,data.shape[1],data.shape[2]))
		calibration[0] += 1.
	else:
		calibration = np.load(fn_cal) ## g,o,v

	print('Calibrating')
	data = prepare.apply_calibration(data,calibration)

	print('Splitting')
	if flag_split == 'L/R':
		dg,dr = prepare.split_lr(data)
	elif flag_split == 'T/B':
		dg,dr = prepare.split_tb(data)
	
	return dg,dr

def get_intensities(dg,dr,spots_g,spots_r,dl,sigma):
	print('Extracting')
	print('Sigma: %.2f'%(sigma))
	print('Box: %dx%d'%(2*dl+1,2*dl+1))
	intensities = np.zeros((spots_g.shape[0], dg.shape[0], 2)) ## NTC
	for i in tqdm(range(intensities.shape[0])):
		ns,bs = ml_psf(dl,dg,sigma,spots_g[i].astype('double'))
		intensities[i,:,0] = ns
		ns,bs = ml_psf(dl,dr,sigma,spots_r[i].astype('double'))
		intensities[i,:,1] = ns

	# intensities = np.zeros((spots_g.shape[0], data.shape[0], 2)) ## NTC
	# for i in range(intensities.shape[0]):
	# 	ox,oy = spots_g[i].astype('int')
	# 	intensities[i,:,0] = d1[:,ox-1:ox+2,oy-1:oy+2].sum((1,2)) - d1[:,ox-1:ox+2,oy-1:oy+2].min((1,2))*9
	# 	ox,oy = spots_r[i].astype('int')
	# 	intensities[i,:,1] = d2[:,ox-1:ox+2,oy-1:oy+2].sum((1,2)) - d2[:,ox-1:ox+2,oy-1:oy+2].min((1,2))*9

	return intensities

def write_hdf5(fn_data,intensities):
	out_dir = spotfinder.get_out_dir(fn_data)
	prefix = os.path.split(out_dir)[1][19:]
	fn_out = os.path.join(out_dir,'intensities_%s.hdf5'%(prefix))
	print('Writing data to: %s'%(fn_out))
	
	try:
		os.remove(fn_out)
	except:
		pass

	f = h5py.File(fn_out,'w')
	f.create_dataset('data',data=intensities,dtype='float64',compression="gzip")
	f.flush()
	f.close()

if __name__ == '__main__':

	flag_split = 'L/R'

	dl = 5
	pixel_real = 6500.
	mag = 60.
	bin = 2.
	lambda_nm = 600.
	NA = 1.2
	motion = 200.
	

	fn_cal = '/Users/colin/Desktop/20240508_caveolin_processing/calibration_individual.npy'
	fn_align = '/Users/colin/Desktop/20240508_caveolin_processing/aligner_results_20240509_caveolin100x_200ms_150mW_1_MMStack_Default/0003_optimize_order2_bin2.theta.npy'
	fn_data = '/Users/colin/Desktop/20240508_caveolin_processing/20240509_caveolin100x_200ms_150mW_1_MMStack_Default.ome.tif'

	# fn_data = '/Users/colin/Desktop/20240508_caveolin_processing/20240509_caveolin100x_50ms_150mW_4_MMStack_Default.ome.tif'

	# fn_data = '/Users/colin/Desktop/riley/20mW_200msFrames_CapA14_24mer_5nMdark4GEprebind-1uM4A-Cy5_inject_1_MMStack_Pos0.ome.tif'
	# fn_align = '/Users/colin/Desktop/riley/aligner_results_20mW_200msFrames_CapA14_24mer_5nMdark4GEprebind-1uM4A-Cy5_inject_1_MMStack_Pos0/0006_optimize_order1_bin1.theta.npy'
	# fn_cal = None

	# fn_data = "/Users/colin/Desktop/projects/test_data/tirf/vc436HS_1_MMStack_Pos0.ome.tif"
	# fn_align = "/Users/colin/Desktop/projects/test_data/tirf/aligner_results_vc436HS_1_MMStack_Pos0/0007_optimize_order3_bin1.theta.npy"
	# fn_cal = None

	# fn_align = '/Users/colin/Desktop/projects/test_data/tirf/aligner_results_5/0004_optimize_order2_bin1.theta.npy'
	# fn_data = '/Users/colin/Desktop/projects/test_data/tirf/5.tif'
	# fn_cal = None


	dg,dr = prepare_data(fn_data,fn_align,fn_cal,flag_split)
	spots_g,spots_r = load_spots(fn_data)
	sigma = calculate_sigma(pixel_real,mag,bin,lambda_nm,NA,motion)
	intensities = get_intensities(dg,dr,spots_g,spots_r,dl,sigma)
	write_hdf5(fn_data,intensities)
	


	









