import os
import time
import shutil
import tifffile
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri

def microscope_expose(npoints,exposure,exposure_properties,properties,fdir,fname='temp',nskip=10):
	try:
		import pycromanager
	except:
		raise Exception('Make sure you have pycromanager installed!!\npip install pycromanager')
	
	## Setup MMCore
	core = pycromanager.Core()

	## Set camera properties
	for prop in properties:
		core.set_property(prop[0],prop[1],prop[2])

	## Set the exposure
	core.set_property(exposure_properties[0],exposure_properties[1],exposure)
	time.sleep(.5) ## Make sure it kicks in
	print('Running', core.get_property(exposure_properties[0],exposure_properties[1]))

	## Start the acquisition - store it in as .tif files that will be removed 
	## Originally wanted to do this completely in memory.... bugs?
	with pycromanager.Acquisition(directory=fdir,name=fname,debug=False,core_log_debug=False,show_display=False) as acq:
		events = pycromanager.multi_d_acquisition_events(num_time_points=npoints, time_interval_s=0.)
		acq.acquire(events)

	## Make sure we close the dataset? 
	dataset = acq.get_dataset()
	path = dataset.path
	dataset.close()

	## Get all the NDTiff files -- large files are split over several individual files
	tifs = [os.path.join(path, tif) for tif in os.listdir(path) if tif.endswith(".tif")]

	## get statistics
	print('Calculating',len(tifs),path)
	mean,var = online_normal_stats(tifs,nskip)

	## Remove the tif data
	shutil.rmtree(path)

	return mean,var

@nb.njit
def collect_t_mins(d,l):
	if d.ndim == 2:
		raise Exception('d should be 3d')
	nt,nx,ny = d.shape
	mt = nt//l

	out = np.zeros((mt,nx,ny),dtype='uint16')
	for i in range(nx):
		for j in range(ny):
			for k in range(mt):
				out[k,i,j] = np.min(d[k*l:(k+1)*l,i,j])
	return out

@nb.njit
def online_expectations(d):
	## the point is the d should stay a uint16 array for memory issues...
	nt,nx,ny = d.shape
	out = np.zeros((2,nx,ny),dtype='double')
	dtij = 0.
	for t in range(nt):
		for i in range(nx):
			for j in range(ny):
				dtij = float(d[t,i,j])
				out[0,i,j] += dtij
				out[1,i,j] += dtij*dtij
	return out

def online_normal_stats(tifs,nskip):
	'''
	take a list of tif filenames and return the mean and var per pixel image of all together
	uses online approx, min-value statistics, and keeps everything uint16 for memory purposes
	'''
	#### Calculate the mean and variance of each pixel, including ALL the tif file parts
	## Note, there are some improvements to make:
	## - the first frame from every file is skipped -- only need to skip it for first file to avoid weird artifacts


	ntifs = len(tifs)
	ns = np.zeros((ntifs))
	for i in range(ntifs):
		## skip the first time point -- only for movie one but idk which that is now (note, not nec. the first filename in list)
		tif = tifs[i]
		d = tifffile.imread(tifs[i])[1:]

		d = collect_t_mins(d,nskip)
		nt,nx,ny = d.shape
		if i == 0:
			expectations = np.zeros((ntifs,2,nx,ny))
		expectations[i] = online_expectations(d)
		ns[i] = nt
	expectations = expectations.sum(0)
	ns = np.sum(ns)
	mean = expectations[0]/ns
	var = expectations[1]/ns - mean**2.

	## use min-value statistics to get actual distribution
	alpha = 0.375
	factor_mean = ndtri((nskip-alpha)/(nskip-2.*alpha+1.))
	a = .85317
	b = -.573889
	factor_var = np.log(nskip)/(a+b/nskip)

	tv = var * factor_var
	tm = mean + factor_mean*np.sqrt(tv)

	return tm,tv

def hamamatsu_setup(fdir,scanmode=3,binning="1x1",maximum_exposure=0.1,nexposures=13):
	'''
	fdir - where to save data
	scanmode - [1,2,3]
	binning - ["1x1","2x2","4x4"]
	maximum_exposure - float (seconds) (light)
	nexposures - number of light expsoures between 100 us and maximum_exposure; log-spaced
	'''

	## This is a temporary file for each individual movie. Make sure the disk has enough space to hold one movie at a time.
	## It will automatically be deleted.
	fname = r'temp'

	## Number of frames to collect -- nb, you really want to have the dark movie dialed in, so use more datapoints there.
	npoints_light = 1000
	npoints_dark  = 5000

	## Automatically set these imaging parameters at the beginning of each acquisition
	properties = [
		['HamamatsuHam_DCAM','ScanMode',3],
		['HamamatsuHam_DCAM','Binning','2x2'],
	]

	## Define the exposures to use. nb, often must be given in msec not sec....
	exposure_properties = ['HamamatsuHam_DCAM','Exposure']
	exposures = 10**np.linspace(-4,np.log10(maximum_exposure),nexposures) * 1000 ## msec
	print('Light Exposures (msec):',exposures)

	dark_exposure = 0.01 * 1000 ## msec
	print('Dark Exposure (msec):',dark_exposure)

	## This script uses min-value order statistics to avoid salt noise. 
	## `nskip` is how many frames to look across when finding each minima.
	nskip = 10

	return fname,npoints_light,npoints_dark,properties,exposure_properties,exposures,dark_exposure,nskip

def run_dark(fdir,fname,npoints_light,npoints_dark,properties,exposure_properties,exposures,dark_exposure,nskip):
	## Collect data, calculate statistics, then remove data
	t0 = time.time()
	mean,var = microscope_expose(npoints_dark,dark_exposure,exposure_properties,properties,fdir,fname,nskip)
	out = np.array([[mean,var],])

	## Save the real data
	t1 = time.time()

	out = np.array(out)
	np.save(os.path.join(fdir,'dark_data.npy'),out)
	print('Done %.3f sec'%(t1-t0))
	print(out.shape)

def run_light(fdir,fname,npoints_light,npoints_dark,properties,exposure_properties,exposures,dark_exposure,nskip):

	out = []

	t0 = time.time()
	for exposure in exposures:
		## Collect data, calculate statistics, then remove data
		mean,var = microscope_expose(npoints_light,exposure,exposure_properties,properties,fdir,fname,nskip)
		out.append([mean,var])

	## Save the real data
	t1 = time.time()
	out = np.array(out)
	np.save(os.path.join(fdir,'light_data.npy'),out)
	np.save(os.path.join(fdir,'light_exposures.npy'),exposures)

	print('Done %.3f sec'%(t1-t0))


def analyze_data(fdir,cut_lower=1400,cut_upper=55000):
	'''
	#### These will be excluded
	cut_lower = 1400   ## the 2x2 binning offet is 400, so ...
	cut_upper = 55000  ## the camera has a 16-bit ADC .... so 65535 max
	'''

	fname_exposures = os.path.join(fdir,'light_exposures.npy')
	fname_light_data = os.path.join(fdir,'light_data.npy')
	fname_dark_data = os.path.join(fdir,'dark_data.npy')

	exposures = np.load(fname_exposures)
	light_data = np.load(fname_light_data)
	dark_data = np.load(fname_dark_data)

	exposures = exposures/1000. ## they are given in msec


	mu = light_data[:,0].mean((1,2))
	last = mu > cut_upper

	keep = np.bitwise_and(mu > cut_lower, mu < cut_upper)
	if np.sum(last) > 0:
		keep[np.argmax(last):] = False
	notkeep = np.bitwise_not(keep)

	if keep.sum() > 0:
		plt.loglog(exposures[keep],mu[keep],'o',color='tab:blue')
	if notkeep.sum() > 0:
		plt.loglog(exposures[notkeep],mu[notkeep],'o',color='tab:red')

	plt.axhline(y=65535,color='black')
	plt.axhspan(ymin = 1, ymax = cut_lower,color='tab:red',alpha=.3)
	plt.axhline(y=cut_lower,color='tab:red')
	plt.axhspan(ymin = cut_upper, ymax = 2**16-1,color='tab:red',alpha=.3)
	plt.axhline(y=cut_upper,color='tab:red')

	plt.ylim(10**2,10**5)
	plt.ylabel('Avg. Camera (DU)')
	plt.xlabel('Exposure Time (s)')
	plt.show()


	## These labels follow Huang et al.
	o = dark_data[0,0]
	var = dark_data[0,1]

	## they use least squares, here is the MLE version 
	g = np.sum((light_data[keep,1] - var[None,:,:])*(light_data[keep,0] - o[None,:,:]),axis=0)/np.sum((light_data[keep,0] - o[None,:,:])**2.,axis=0)


	fig,ax = plt.subplots(1,2,figsize=(7,3),dpi=300)
	hy,hx = ax[0].hist(g.flatten(),bins=500,log=True,range=(0,10),color='tab:blue')[:2]
	ax[0].hist(g[128:,:-128].flatten(),bins=500,log=True,histtype='step',color='tab:orange')
	ax[1].imshow(g,vmin=4,vmax=6)
	ax[0].set_xlabel('g (DU/e-)')
	ax[0].set_ylabel('Pixels')
	ax[1].axis('off')
	fig.tight_layout()
	plt.show()

	peak_g = hx[hy.argmax()]
	print('Maximum:',peak_g)


	fig,ax = plt.subplots(1,2,figsize=(7,3),dpi=300)
	hy,hx = ax[0].hist(o.flatten(),bins=500,log=True,color='tab:blue')[:2]
	ax[0].hist(o[128:,:-128].flatten(),bins=500,log=True,histtype='step',color='tab:orange')
	ax[1].imshow(o,vmin=380,vmax=420)
	ax[0].set_xlabel('o (DU)')
	ax[0].set_ylabel('Pixels')
	ax[1].axis('off')
	fig.tight_layout()
	plt.show()

	peak_o = hx[hy.argmax()]
	print('Maximum:',peak_o)



	fig,ax = plt.subplots(1,2,figsize=(7,3),dpi=300)
	hy,hx = ax[0].hist(var.flatten(),bins=500,log=True,range=(0,1000),color='tab:blue')[:2]
	ax[0].hist(var[128:,:-128].flatten(),bins=500,log=True,histtype='step',color='tab:orange')
	ax[1].imshow(var,vmin=50,vmax=400,interpolation='nearest')
	ax[0].set_xlabel(r'var ($DU^2$)')
	ax[0].set_ylabel('Pixels')
	ax[1].axis('off')
	fig.tight_layout()
	plt.show()

	peak_var = hx[hy.argmax()]
	print('Maximum:',peak_var)
	print('Noisy electrons:',np.sqrt(peak_var)/peak_g)


	### Save the calibration
	print('Saving calibrations (individual and global)')
	print('g:   %.3f'%(peak_g))
	print('o:   %.3f'%(peak_o))
	print('var: %.3f'%(peak_var))

	out = np.array([g,o,var,])
	np.save(os.path.join(fdir,'calibration_individual.npy'),out)

	out = np.array([g*0.+peak_g,o*0.+peak_o,var*0+peak_var])
	np.save(os.path.join(fdir,'calibration_global.npy'),out)


	fig,ax = plt.subplots(1,2,figsize=(7,3))

	avgphotons = (light_data[:,0]-o)/g
	lam = np.median((np.median(avgphotons,axis=(1,2))/exposures)[:-1])
	ax[0].loglog(exposures[:-1],(np.mean(avgphotons,axis=(1,2))/exposures)[:-1],'o',color='tab:blue',label='Data')
	ax[0].axhline(y=lam,color='tab:orange',label='Median')
	ax[0].set_xlabel('Exposure (s)')
	ax[0].set_ylabel('Avg. Photon rate (s^-1)')
	ax[0].legend()

	t = 10**np.linspace(np.log10(exposures[0])//1-1,np.log10(exposures[-1])//1+1,1000)
	yy = g.mean()*lam*t+o.mean()
	yy[yy>=2**16-1] = 2**16-1
	ax[1].loglog(t,yy,label='Model',color='tab:orange')
	ax[1].loglog(exposures,light_data[:,0].mean((1,2)),'o',label='Data',color='tab:blue')
	ax[1].legend()

	ax[1].axhline(y=cut_lower,color='tab:red')
	ax[1].axhline(y=cut_upper,color='tab:red')

	ax[1].set_ylabel('Avg. Camera Counts (DU)')
	ax[1].set_xlabel('Exposure (s)')
	fig.tight_layout()
	plt.show()

def help():
	msg = r'''import highfret

fdir = r'C:\Users\ptirf\Desktop'

setup = highfret.calibrater.hamamatsu_setup(fdir,scanmode=2,binning="2x2",maximum_exposure=0.1)
highfret.calibrater.run_dark(fdir,*setup)
highfret.calibrater.run_light(fdir,*setup)
highfret.calibrater.analyze_data(fdir,cut_lower=1400,cut_upper=55000)'''
	print(msg)