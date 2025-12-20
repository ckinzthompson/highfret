import numpy as np
from tqdm import tqdm
from scipy.special import erf
import matplotlib.pyplot as plt

from . import shape_evidence

def calculate_sigma(pixel_real,mag,bin,lambda_nm,NA,motion):
	pixel_xy = pixel_real/mag*bin
	sigma_xy = 0.21*lambda_nm/NA
	sigma_xy = np.sqrt(motion**2.+sigma_xy**2.)
	sigma = sigma_xy/pixel_xy
	return sigma

def sweep_sigma_lnev(movie, spots, first: int=0, last: int=0, optmethod: str='ACF',sigma_low: float=0.2, sigma_high: float=2.0, nsigma: int=61, dl: int=5,):
	nt,nx,ny = movie.shape

	if last == 0:
		last = int(nt)
	nt = last-first

	if not optmethod is None:
		if optmethod.lower() in ['max','mean','std','acf']:
			nt = 1

	sigmas = np.logspace(np.log10(sigma_low),np.log10(sigma_high),nsigma)
	all_sigs = np.zeros((spots.shape[0],nt),dtype='int')


	for ni in tqdm(range(spots.shape[0])):
		xyi = spots[ni]

		xmin = int(max(0,xyi[0]-dl))
		xmax = int(min(movie.shape[1]-1,xyi[0]+dl) + 1)
		ymin = int(max(0,xyi[1]-dl))
		ymax = int(min(movie.shape[2]-1,xyi[1]+dl) + 1)

		gx,gy = np.mgrid[xmin:xmax,ymin:ymax]
		gx = gx.astype('f').flatten()
		gy = gy.astype('f').flatten()

		data = movie[first:last,xmin:xmax,ymin:ymax].astype('f').reshape((last-first,gx.size)).copy()

		if not optmethod is None:
			if optmethod.lower() == 'max':
				data = data.max(0)[None,:]
			elif optmethod.lower() == 'mean':
				data = data.mean(0)[None,:]
			elif optmethod.lower() == 'std':
				data = data.std(0)[None,:]
			elif optmethod.lower() == 'acf':
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

