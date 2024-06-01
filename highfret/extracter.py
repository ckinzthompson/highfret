import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import os
import h5py
import numpy as np
import numba as nb
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from . import prepare,minmax,alignment,spotfinder,shape_evidence

from scipy.special import erf

def mask_extract(l,movie,sigma,xyi,maxiters=1000):
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
		mask = psi.argsort()[-4:]
		b = np.median(m,axis=1)
		n = ((m-b[:,None])[:,mask]).sum(1)

	except:
		n = np.zeros(movie.shape[0])
		b = np.zeros(movie.shape[0])

	return n,b


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

		mask = psi.argsort()[-4:]
		b = np.median(m,axis=1)
		n = ((m-b[:,None])[:,mask]).sum(1)

		n0 = np.mean(b)
		psum = np.sum(psi**2.)
		npm = np.median
		# npm = np.mean

		for iter in range(maxiters):
			b = npm(m - n[:,None]*psi[None,:],axis=(1)) ## median is much more stable
			n = np.sum((m - b[:,None])*psi[None,:],axis=(1))/psum
			n1 = np.mean(b)

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

def sweep_sigma_lnev(movie,spots,l=5,nsigma=61,sigma_low=.2,sigma_high=2.,method='acf',first=0,last=0):
	nt,nx,ny = movie.shape

	if last == 0:
		last = int(nt)
	nt = last-first

	if not method is None:
		if method.lower() in ['max','mean','std','acf']:
			nt = 1

	sigmas = np.logspace(np.log10(sigma_low),np.log10(sigma_high),nsigma)
	all_sigs = np.zeros((spots.shape[0],nt),dtype='int')


	for ni in tqdm(range(spots.shape[0])):
		xyi = spots[ni]

		xmin = int(max(0,xyi[0]-l))
		xmax = int(min(movie.shape[1]-1,xyi[0]+l) + 1)
		ymin = int(max(0,xyi[1]-l))
		ymax = int(min(movie.shape[2]-1,xyi[1]+l) + 1)

		gx,gy = np.mgrid[xmin:xmax,ymin:ymax]
		gx = gx.astype('f').flatten()
		gy = gy.astype('f').flatten()

		data = movie[first:last,xmin:xmax,ymin:ymax].astype('f').reshape((last-first,gx.size)).copy()

		if not method is None:
			if method.lower() == 'max':
				data = data.mean(0)[None,:]
			elif method.lower() == 'mean':
				data = data.mean(0)[None,:]
			elif method.lower() == 'std':
				data = data.mean(0)[None,:]
			elif method.lower() == 'acf':
				dmean = data.mean(0)[None,:]
				data = (np.nanmean((data[1:]-dmean)*(data[:-1]-dmean),axis=0)/np.nanmean((data-dmean)**2.,axis=0))[None,:]

		lnevs = np.zeros((data.shape[0],sigmas.size+1))
		templates = np.zeros((sigmas.size,data.shape[1])) ## flat!!!!
		for i in range(sigmas.size):
			sigma = sigmas[i]
			dex = .5 * (erf((xyi[0]-gx+.5)/(np.sqrt(2.*sigma**2.))) - erf((xyi[0]-gx -.5)/(np.sqrt(2.*sigma**2.))))
			dey = .5 * (erf((xyi[1]-gy+.5)/(np.sqrt(2.*sigma**2.))) - erf((xyi[1]-gy -.5)/(np.sqrt(2.*sigma**2.))))
			psi = dex*dey
			psi /= psi.sum()
			templates[i] = psi

			for t in range(data.shape[0]):
				lnevs[t,i] = shape_evidence.ln_evidence(templates[i],data[t])
				if i == 0:
					lnevs[t,-1] = shape_evidence.null_ln_evidence(data[t])
		y = lnevs.argmax(1)
		all_sigs[ni] = y
	return sigmas, all_sigs


def optimize_sigma(fn_data,fn_align,fn_cal,flag_split,l=5,nsigma=61,sigma_low=.2,sigma_high=2.,flag_keeptbbins=False,method='acf',first=0,last=0):

	dg,dr = prepare_data(fn_data,fn_align,fn_cal,flag_split)
	spots_g,spots_r = load_spots(fn_data)

	fig,ax = plt.subplots(1)

	print('Optimizing sigma')

	for movie,spots,color,label in zip([dg,dr],[spots_g,spots_r],['tab:green','tab:red'],['Green','Red']):
		sigmas,all_sigs = sweep_sigma_lnev(movie,spots,l,nsigma,sigma_low,sigma_high,method,first,last)

		if not flag_keeptbbins:
			keep = np.bitwise_and(all_sigs>0,all_sigs<nsigma-1) ## remove first, last, and null
		else:
			keep = all_sigs < nsigma 
		bins = np.append(sigmas,sigmas[-1]*1.1)
		good = np.array([sigmas[gi] for gi in all_sigs[keep]])

		median_sigma = np.median(good)*(1+1./nsigma) ## ~middle of bin
		print("%s: Median sigma (px): %.3f"%(label,median_sigma))

		hy,hx = ax.hist(good,bins=bins,color=color,alpha=.5)[:2]
		max_sigma = (.5*(hx[:-1]+hx[1:]))[hy.argmax()] ## middle of bin
		print("%s: Maximum sigma (px): %.3f"%(label,max_sigma))

		ax.axvline(median_sigma,color=color)
		ax.axvline(max_sigma,color=color,ls='--')
		ax.set_xlabel(r'Empirical (shaped) PSF $\sigma$ (px)')
		ax.set_ylabel('Counts')
	return fig, ax


def load_spots(fn_data):
	print('Loading Spots')
	
	dirs = prepare.get_out_dir(fn_data)
	dir_top = dirs[0]
	dir_temp = dirs[1]
	dir_extracter = dirs[4]
	
	fn_spots_g = os.path.join(dir_temp,'g_spots.npy')
	fn_spots_r = os.path.join(dir_temp,'r_spots.npy')
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

def get_intensities(dg,dr,spots_g,spots_r,dl,sigma,method_fxn='MLE PSF'):
	if method_fxn == 'MLE PSF':
		method_fxn = ml_psf
	else:
		method_fxn = mask_extract

	print('Extracting')
	print('Sigma: %.2f'%(sigma))
	print('Box: %dx%d'%(2*dl+1,2*dl+1))
	intensities = np.zeros((spots_g.shape[0], dg.shape[0], 2)) ## NTC
	
	for i in tqdm(range(intensities.shape[0])):
		ns,bs = method_fxn(dl,dg,sigma,spots_g[i].astype('double'))
		intensities[i,:,0] = ns
		ns,bs = method_fxn(dl,dr,sigma,spots_r[i].astype('double'))
		intensities[i,:,1] = ns

	return intensities

def write_hdf5(fn_data,intensities):
	dirs = prepare.get_out_dir(fn_data)
	dir_top = dirs[0]
	dir_spotfinder = dirs[3]
	dir_extracter = dirs[4]

	prefix = os.path.split(dir_top)[1][9:]
	# fn_out = os.path.join(out_dir,'intensities_%s.hdf5'%(prefix))
	fn_out = os.path.join(dir_extracter,'trajectories.hdf5')
	print('Writing data to: %s'%(fn_out))
	
	try:
		os.remove(fn_out)
	except:
		pass

	f = h5py.File(fn_out,'w')
	f.create_dataset(prefix, data=intensities, dtype='float64',compression="gzip")
	f.flush()
	f.close()


	









