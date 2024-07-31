import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import os
import h5py
import numpy as np
import numba as nb
from tqdm import tqdm
import matplotlib.pyplot as plt

from . import prepare,minmax,alignment,spotfinder,shape_evidence,mle_extract

from scipy.special import erf

def default_flags():
	df = {
		'split':'L/R',
		'method':'MLE PSF',
		'dl':5,
		'sigma':.8,

		'nsigma':61,
		'sigma_low':.2,
		'sigma_high':2.,
		'keeptbbins':False,
		'optmethod':'ACF',
		'first':0,
		'last':0,

		'pixel_real':6500.,
		'mag':60.,
		'bin':2.,
		'lambda_nm':580.,
		'NA':1.2,
		'motion':100,
		
		'median_filter':0,
		'neighbors':True,
		'max_restarts':20,
		'correct':False,
		# 'cutoff_rms':1.0,
		'cutoff_rel':1e-3,
	}
	return df

def safe_flags(flags):
	df = default_flags()
	for key in df.keys():
		if not key in flags:
			flags[key] = df[key]
	return flags 

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

def prepare_data(fn_data,fn_align,fn_cal,flags=default_flags()):
	flags = safe_flags(flags)

	theta = np.load(fn_align)
	data = prepare.load(fn_data)
	if fn_cal is None:
		calibration = np.zeros((3,data.shape[1],data.shape[2]))
		calibration[0] += 1.
	else:
		calibration = np.load(fn_cal) ## g,o,v

	print('Calibrating')
	prepare.apply_calibration(data,calibration) ## preparation is in place

	print('Splitting')
	if flags['split'] == 'L/R':
		dg,dr = prepare.split_lr(data)
	elif flags['split'] == 'T/B':
		dg,dr = prepare.split_tb(data)

	if flags['median_filter'] > 0:
		print('Filtering')
		dg = prepare.median_scmos(dg,int(flags['median_filter']))
		dr = prepare.median_scmos(dr,int(flags['median_filter']))

	return dg,dr

def calculate_sigma(pixel_real,mag,bin,lambda_nm,NA,motion):
	pixel_xy = pixel_real/mag*bin
	sigma_xy = 0.21*lambda_nm/NA
	sigma_xy = np.sqrt(motion**2.+sigma_xy**2.)
	sigma = sigma_xy/pixel_xy
	return sigma

def sweep_sigma_lnev(movie,spots,flags=default_flags()):
	flags = safe_flags(flags)
	nt,nx,ny = movie.shape

	if flags['last'] == 0:
		flags['last'] = int(nt)
	nt = flags['last']-flags['first']

	if not flags['optmethod'] is None:
		if flags['optmethod'].lower() in ['max','mean','std','acf']:
			nt = 1

	sigmas = np.logspace(np.log10(flags['sigma_low']),np.log10(flags['sigma_high']),flags['nsigma'])
	all_sigs = np.zeros((spots.shape[0],nt),dtype='int')


	for ni in tqdm(range(spots.shape[0])):
		xyi = spots[ni]

		xmin = int(max(0,xyi[0]-flags['dl']))
		xmax = int(min(movie.shape[1]-1,xyi[0]+flags['dl']) + 1)
		ymin = int(max(0,xyi[1]-flags['dl']))
		ymax = int(min(movie.shape[2]-1,xyi[1]+flags['dl']) + 1)

		gx,gy = np.mgrid[xmin:xmax,ymin:ymax]
		gx = gx.astype('f').flatten()
		gy = gy.astype('f').flatten()

		data = movie[flags['first']:flags['last'],xmin:xmax,ymin:ymax].astype('f').reshape((flags['last']-flags['first'],gx.size)).copy()

		if not flags['optmethod'] is None:
			if flags['optmethod'].lower() == 'max':
				data = data.max(0)[None,:]
			elif flags['optmethod'].lower() == 'mean':
				data = data.mean(0)[None,:]
			elif flags['optmethod'].lower() == 'std':
				data = data.std(0)[None,:]
			elif flags['optmethod'].lower() == 'acf':
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


def optimize_sigma(fn_data,fn_align,fn_cal,flags=default_flags()):
	flags = safe_flags(flags)

	dg,dr = prepare_data(fn_data,fn_align,fn_cal,flags)
	spots_g,spots_r = load_spots(fn_data)

	fig,ax = plt.subplots(1)

	print('Optimizing sigma')

	record = {'median':[],'maximum':[]}

	for movie,spots,color,label in zip([dg,dr],[spots_g,spots_r],['tab:green','tab:red'],['Green','Red']):
		sigmas,all_sigs = sweep_sigma_lnev(movie,spots,flags)

		if not flags['keeptbbins']:
			keep = np.bitwise_and(all_sigs>0,all_sigs<flags['nsigma']-1) ## remove first, last, and null
		else:
			keep = all_sigs < nsigma 
		bins = np.append(sigmas,sigmas[-1]*1.1)
		good = np.array([sigmas[gi] for gi in all_sigs[keep]])

		median_sigma = np.median(good)*(1+1./flags['nsigma']) ## ~middle of bin
		record['median'].append(median_sigma)
		print("%s: Median sigma (px): %.3f"%(label,median_sigma))

		hy,hx = ax.hist(good,bins=bins,color=color,alpha=.5)[:2]
		maximum_sigma = (.5*(hx[:-1]+hx[1:]))[hy.argmax()] ## middle of bin
		record['maximum'].append(maximum_sigma)
		print("%s: Maximum sigma (px): %.3f"%(label,maximum_sigma))

		ax.axvline(median_sigma,color=color)
		ax.axvline(maximum_sigma,color=color,ls='--')
		ax.set_xlabel(r'Empirical (shaped) PSF $\sigma$ (px)')
		ax.set_ylabel('Counts')
	return fig, ax, record

def correct_NB(NB):
	## correct the offset (due to bad order statistics) with a common value to all
	dmin = NB[0].min(1) ## across time
	mean = np.median(dmin)
	ul = np.percentile(dmin,[5.,95.])
	keep = np.bitwise_and(dmin>=ul[0],dmin<=ul[1])
	var = np.var(dmin[keep])
	from scipy.special import ndtri
	nn = NB.shape[2] ## time
	## use min-value statistics to get actual distribution
	alpha = 0.375
	factor_mean = ndtri((nn-alpha)/(nn-2.*alpha+1.))
	a = .85317
	b = -.573889
	factor_var = np.log(nn)/(a+b/nn)
	tv = var * factor_var
	tm = mean + factor_mean*np.sqrt(tv)
	## apply correction
	NB[0] -= tm
	NB[1] += tm/4.
	return NB

@nb.njit(cache=True)
def vanilla(subdata):
	nt,nx,ny = subdata.shape
	NB = np.zeros((2,nt))
	ss = nx*ny ## sometimes edge spots are smaller than 3x3

	for t in range(nt):
		sorted = np.sort(subdata[t].flatten())
		suml = float(sorted[0]) + float(sorted[1]) + float(sorted[2])
		sumh = float(sorted[ss-1]) + float(sorted[ss-2]) + float(sorted[ss-3]) + float(sorted[ss-4])
		NB[1,t] = suml/3. ## bg is average non-top pixel; by excluding non-spot pixels, you have a order statistics for remaining. these cancel.
		NB[0,t] = sumh-4.*NB[1,t] ## each pixel has a bg contribution we need to substract out
		
	return NB

def extract_mle_all(data,spots,flags):
	flags = safe_flags(flags)

	## N,B are (nmol,nt)
	## data is (t,x,y)
	nsigma = 5
	offset = 0.5
	
	grid = mle_extract.gridclass(np.array([0.,0.]),np.array([1.,1.]),np.array(data[0].shape))
	nmol = spots.shape[0]
	nt = data.shape[0]

	guess = extract_vanilla(data,spots,flags)
	NB0 = guess.copy()
	# N0 = np.array([data[:,int(spotsi[0]),int(spotsi[1])] for spotsi in spots]).astype('double')
	# B0 = N0*0 + np.mean(data,axis=(1,2))[None,:].astype('double')
	# N0 -= B0
	# NB0 = np.array([N0,B0])

	## Neighbor distances
	distance = np.sqrt((spots[:,None,0]-spots[None,:,0])**2. + (spots[:,None,1]-spots[None,:,1])**2.)
	cutoff = np.sqrt(2.)*flags['dl'] + nsigma*flags['sigma']
	if flags['neighbors']:
		keep_neighbors = distance < cutoff
	else:
		keep_neighbors = distance == 0.

	## loop calculations 
	rms = [np.nan,]
	rel = np.nan
	for iterations in range(flags['max_restarts']):
		NB1 = NB0.copy()
		for moli in tqdm(range(nmol),desc=f'Iter: {iterations}, RMSD: {rms[-1]:.1f}, Rel.: {rel:.3f}'):
			subgrid = mle_extract.subgrid_center(grid, spots[moli], flags['dl'])
			subdata = mle_extract.subgrid_extract(grid, data, subgrid).astype('double')		
			NB1[:,moli] = mle_extract.mle_all(
				subdata,
				subgrid.origin,
				subgrid.dxy,
				subgrid.nxy,
				spots,
				keep_neighbors,
				moli,
				NB0,
				flags['sigma'],
				nsigma,
				offset)
		rms.append(np.sqrt(np.mean(np.square(guess[0] - NB1[0]))))

		if iterations > 0:
			rel = np.abs(rms[-1]-rms[-2])/rms[-2]

		# if rms[-1] < flags['cutoff_rms'] or rel < flags['cutoff_rel']:
		if rel < flags['cutoff_rel']:
			print(f'Done! Final RMSD: {rms[-1]:.1f}, rel.: {rel:.6f}')
			break
		NB0 = NB1.copy()

	return NB1

def extract_vanilla(data,spots,flags):
	flags = safe_flags(flags)

	## N,B are (nmol,nt)
	## data is (t,x,y)
	# nsigma = 5
	# offset = 0.5
	
	grid = mle_extract.gridclass(np.array([0.,0.]),np.array([1.,1.]),np.array(data[0].shape))
	nmol = spots.shape[0]
	nt = data.shape[0]

	NB = np.zeros((2,nmol,nt),dtype='double')
	
	## loop calculations 
	for moli in tqdm(range(nmol),desc=f'Vanilla'):
		subgrid = mle_extract.subgrid_center(grid,spots[moli],1)
		subdata = mle_extract.subgrid_extract(grid,data,subgrid)
		NB[:,moli] = vanilla(subdata)

	if flags['correct']:
		correct_NB(NB)

	return NB

def extract_wrapper(dg,dr,spots_g,spots_r,flags=default_flags()):
	flags = safe_flags(flags)

	if flags['method'] == 'MLE PSF':
		method_fxn = extract_mle_all
	elif flags['method'] == 'Max Px':
		method_fxn = extract_vanilla
	else:
		raise Exception('{method} not implemented')

	print('Extracting')
	print('Sigma: %.2f'%(flags['sigma']))
	print('Box: %dx%d'%(2*flags['dl']+1,2*flags['dl']+1))

	intensities = np.zeros((spots_g.shape[0], dg.shape[0], 2)) ## NTC
	print('Green:')
	intensities[:,:,0] = method_fxn(dg,spots_g,flags)[0]
	print('Red:')
	intensities[:,:,1] = method_fxn(dr,spots_r,flags)[0]
	
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
	
def run_job_optimize(job):
	fn_data = job['fn_data']
	fn_align = job['fn_align']
	fn_cal = job['fn_cal']
	flags = job
	
	dirs = prepare.get_out_dir(fn_data)
	from .spotfinder import safe_align
	fn_align = safe_align(fn_data,fn_align)

	dir_extracter = dirs[4]

	fig,ax,record = optimize_sigma(fn_data,fn_align,fn_cal,flags)
	[plt.savefig(os.path.join(dir_extracter,'sigma_optimization.%s'%(ext))) for ext in ['png','pdf']]

	for key in record.keys():
		job[key] = record[key]

	prepare.dump_job(os.path.join(dir_extracter,'job_optimize.txt'),'Job Name: Optimize PSF Sigma',job)

	return fig

def run_job_extract(job):
	fn_data = job['fn_data']
	fn_align = job['fn_align']
	fn_cal = job['fn_cal']
	flags = job

	dirs = prepare.get_out_dir(fn_data)
	from .spotfinder import safe_align
	fn_align = safe_align(fn_data,fn_align)

	dir_extracter = dirs[4]
	
	dg,dr = prepare_data(fn_data,fn_align,fn_cal,flags)
	spots_g,spots_r = load_spots(fn_data)
	intensities = extract_wrapper(dg,dr,spots_g,spots_r,flags)
	
	write_hdf5(fn_data,intensities)

	fig,ax = plt.subplots(1)
	ax.plot(np.nanmean(intensities,axis=0)[:,0],color='tab:green',lw=1)
	ax.plot(np.nanmean(intensities,axis=0)[:,1],color='tab:red',lw=1)
	ax.set_xlabel('Time (frame)')
	ax.set_ylabel('Average Intensity')
	fig.set_figheight(6.)
	fig.set_figwidth(6.)
	[plt.savefig(os.path.join(dir_extracter,'intensity_avg.%s'%(ext))) for ext in ['png','pdf']]

	prepare.dump_job(os.path.join(dir_extracter,'job_extract.txt'),'Job Name: Extract Intensities',job)

	return fig





