import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import numpy as np
import numba as nb
from tqdm import tqdm

from . import mle_extract

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
def subvanilla(subdata):
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

def extract_vanilla(data, spots, correct: bool = False, verbose: bool = True):
	## N,B are (nmol,nt)
	## data is (t,x,y)
	
	grid = mle_extract.gridclass(np.array([0.,0.]),np.array([1.,1.]),np.array(data[0].shape))
	nmol = spots.shape[0]
	nt = data.shape[0]

	NB = np.zeros((2,nmol,nt),dtype='double')
	
	## loop calculations 
	loop = range(nmol)
	if verbose:
		loop = tqdm(loop,desc=f'Vanilla')
	for moli in loop:
		subgrid = mle_extract.subgrid_center(grid,spots[moli],1)
		subdata = mle_extract.subgrid_extract(grid,data,subgrid)
		NB[:,moli] = subvanilla(subdata)

	if correct:
		correct_NB(NB)

	return NB

def extract_mle(data, spots, sigma:float=0.8, dl:int=5, neighbors:bool=True,max_restarts:int=20, cutoff_rel:float=1e-3, correct:bool=False, verbose:bool=True):
	## N,B are (nmol,nt)
	## data is (t,x,y)
	nsigma = 5
	offset = 0.5
	
	grid = mle_extract.gridclass(np.array([0.,0.]),np.array([1.,1.]),np.array(data[0].shape))
	nmol = spots.shape[0]
	nt = data.shape[0]

	guess = extract_vanilla(data,spots,correct,verbose)
	NB0 = guess.copy()
	# N0 = np.array([data[:,int(spotsi[0]),int(spotsi[1])] for spotsi in spots]).astype('double')
	# B0 = N0*0 + np.mean(data,axis=(1,2))[None,:].astype('double')
	# N0 -= B0
	# NB0 = np.array([N0,B0])

	## Neighbor distances
	distance = np.sqrt((spots[:,None,0]-spots[None,:,0])**2. + (spots[:,None,1]-spots[None,:,1])**2.)
	cutoff = np.sqrt(2.)*dl + nsigma*sigma
	if neighbors:
		keep_neighbors = distance < cutoff
	else:
		keep_neighbors = distance == 0.

	## loop calculations 
	rms = [np.nan,]
	rel = np.nan
	for iterations in range(max_restarts):
		NB1 = NB0.copy()
		for moli in tqdm(range(nmol),desc=f'Iter: {iterations}, RMSD: {rms[-1]:.1f}, Rel.: {rel:.3f}'):
			subgrid = mle_extract.subgrid_center(grid, spots[moli], dl)
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
				sigma,
				nsigma,
				offset)
		rms.append(np.sqrt(np.mean(np.square(guess[0] - NB1[0]))))

		if iterations > 0:
			rel = np.abs(rms[-1]-rms[-2])/rms[-2]

		# if rms[-1] < flags['cutoff_rms'] or rel < flags['cutoff_rel']:
		if rel < cutoff_rel:
			break
		NB0 = NB1.copy()

	return NB1,rms[-1],rel
