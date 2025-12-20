import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .containers import general_analysis_class
from .support import psf
from .support.extracter import extract_vanilla as _extract_vanilla
from .support.extracter import extract_mle as _extract_mle


def extract_vanilla(analysis: general_analysis_class, sigma:float=0.8, dl:int=5, correct:bool=False, verbose:bool=True):
	analysis.log += 'Extracting (Vanilla)\n'
	analysis.log += f'\tSigma: {sigma:.2f}\n'
	analysis.log += f'\tBox: {2*dl+1}x{2*dl+1}\n'
	
	t0 = time.time()
	analysis.traces = np.zeros((analysis.spots.shape[1], analysis.data.shape[1], analysis.data.shape[0])) ## NTC
	for i in range(analysis.data.shape[0]):
		NB = _extract_vanilla(analysis.data[i], analysis.spots[i], correct, verbose)
		analysis.traces[:,:,i] = NB[0] ## 0 is N, 1 is B
	t1 = time.time()
	analysis.log += f'\tFinished in {t1-t0:.3f} s\n'

def extract_mle(analysis: general_analysis_class, sigma:float=0.8, dl:int=5, neighbors:bool=True,max_restarts:int=20, cutoff_rel:float=1e-3, correct:bool=False, verbose:bool=True):
	analysis.log += 'Extracting (MLE)\n'
	analysis.log += f'\tSigma: {sigma:.2f}\n'
	analysis.log += f'\tBox: {2*dl+1}x{2*dl+1}\n'

	t0 = time.time()
	analysis.traces = np.zeros((analysis.spots.shape[1], analysis.data.shape[1], analysis.data.shape[0])) ## NTC
	for i in range(analysis.data.shape[0]):
		NB,rms,rel = _extract_mle(analysis.data[i], analysis.spots[i], sigma=sigma, dl=dl, neighbors=neighbors,max_restarts=max_restarts,cutoff_rel=cutoff_rel,correct=correct, verbose=verbose) 
		analysis.traces[:,:,i] = NB[0] ## 0 is N, 1 is B
		analysis.log += f'\tColor {i}: final RMSD = {rms:.1f}, rel.= {rel:.6f}\n'
	t1 = time.time()
	analysis.log += f'\tFinished in {t1-t0:.3f} s\n'

def optimize_sigma(analysis: general_analysis_class, first:int=0, last:int=0, optmethod:str='ACF', sigma_low:float=0.2, sigma_high:float=2.0, nsigma:int=61, dl:int=5, keeptbbins:bool=False):
	analysis.log += 'Optimizing Sigma\n'
	ncolors = analysis.data.shape[0]

	record_median = []
	record_maximum = []

	color_cycle = ['tab:green', 'tab:red']
	fig,ax = plt.subplots(1)
	for i in range(ncolors):
		movie = analysis.data[i]
		spots = analysis.spots[i]
		label = f'Color {i}'

		sigmas,all_sigs = psf.sweep_sigma_lnev(movie, spots, first, last, optmethod, sigma_low, sigma_high, nsigma, dl)
		sigmas /= np.sqrt(2.) ## b/c they're fitting the ACF image not the image. all_sigs is indices to sigmas...

		if keeptbbins:
			keep = np.bitwise_and(all_sigs>0,all_sigs<nsigma-1) ## remove first, last, and null
		else:
			keep = all_sigs < nsigma 
		bins = np.append(sigmas,sigmas[-1]*1.1)
		good = np.array([sigmas[gi] for gi in all_sigs[keep]])

		median_sigma = np.median(good)*(1+1./nsigma) ## ~middle of bin
		record_median.append(median_sigma)
		analysis.log += f'\t{label}: Median sigma (px): {median_sigma:.3f}\n'

		if i < len(color_cycle):
			hy,hx = ax.hist(good,bins=bins,alpha=.5,color=color_cycle[i])[:2]
		else:
			hy,hx = ax.hist(good,bins=bins,alpha=.5)[:2]
		color = ax.patches[-1].get_facecolor()
		maximum_sigma = (.5*(hx[:-1]+hx[1:]))[hy.argmax()] ## middle of bin
		record_maximum.append(maximum_sigma)
		analysis.log += f'\t{label}: Maximum sigma (px): {maximum_sigma:.3f}\n'

		ax.axvline(median_sigma,color=color)
		ax.axvline(maximum_sigma,color=color,ls='--')
	ax.set_xlabel(r'Empirical (shaped) PSF $\sigma$ (px)')
	ax.set_ylabel('Counts')

	# fdir = Path(fdir)
	# [plt.savefig(fdir / f'sigma_optimization.{ext}') for ext in ['png','pdf']]

	return fig, ax, record_median, record_maximum

def figure_avg_intensity(analysis: general_analysis_class):
	color_cycle = ['tab:green', 'tab:red']
	fig,ax = plt.subplots(1)
	for i in range(analysis.traces.shape[2]):
		if i < len(color_cycle):
			ax.plot(np.nanmean(analysis.traces[:,:,i],axis=0),lw=1,color=color_cycle[i])
		else:
			ax.plot(np.nanmean(analysis.traces[:,:,i],axis=0),lw=1)
	ax.set_xlabel('Time (frame)')
	ax.set_ylabel('Average Intensity')
	fig.set_figheight(6.)
	fig.set_figwidth(6.)

	# [plt.savefig(fdir / f'intensity_avg.{ext}') for ext in ['png','pdf']]
	return fig,ax