import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import numpy as np
import numba as nb
from scipy.ndimage import gaussian_filter

from . import minmax, punch
from . import modelselect_alignment as alignment

@nb.njit(cache=True)
def compare_close_spots(spots1,spots2,cutoff):
	### replace close spots with their average position
	kept = 0
	ns1 = spots1.shape[0]
	ns2 = spots2.shape[0]
	out1 = np.zeros((ns1,2))
	out2 = np.zeros((ns2,2))

	already_found1 = np.zeros(ns1,dtype='int')
	already_found2 = np.zeros(ns2,dtype='int')

	for i in range(ns1):
		if already_found1[i] == 1:
			continue 
		for j in range(ns2):
			if already_found2[j] == 1:
				continue

			r_ij = np.sqrt((float(spots1[i,0])-float(spots2[j,0]))**2.+(float(spots1[i,1])-float(spots2[j,1]))**2.)
			if r_ij < cutoff:
				already_found1[i] = 1
				already_found2[j] = 1
				out1[kept] = spots1[i]
				out2[kept] = spots2[j]
				kept +=1
				break
	return out1[:kept],out2[:kept]

@nb.njit(cache=True)
def remove_close_spots(spots,cutoff):
	### replace close spots with their average position
	kept = 0
	ns = spots.shape[0]
	out = np.zeros_like(spots)

	already_found = np.zeros(ns,dtype='int')

	for i in range(ns):
		if already_found[i] == 1:
			continue 
		for j in range(i+1,ns):
			r_ij = np.sqrt((float(spots[i,0])-float(spots[j,0]))**2.+(float(spots[i,1])-float(spots[j,1]))**2.)
			if r_ij < cutoff:
				already_found[i] = 1
				already_found[j] = 1
				out[kept,0] = (float(spots[i,0])+float(spots[j,0]))/2.
				out[kept,1] = (float(spots[i,1])+float(spots[j,1]))/2.
				kept +=1
				break
		if already_found[i] == 0:
			out[kept,0] = float(spots[i,0])
			out[kept,1] = float(spots[i,1])
			kept += 1
	return out[:kept]

@nb.njit(cache=True)
def find_outofframe(img,spots):
	nx,ny = img.shape
	ns,_ = spots.shape
	keep = np.ones(ns,dtype='bool')

	for i in range(ns):
		if spots[i,0] < 0 or spots[i,0] >= nx:
			keep[i] = False
		if spots[i,1] < 0 or spots[i,1] >= ny:
			keep[i] = False
	return keep

def refine_simple(img, spots, l=2, max_shift=1.):
	punches = punch.get_punches(img, spots, l=l, fill_value=np.nan)
	
	x = np.arange(punches.shape[1]).astype('double')
	x -= float(x.size//2) + 0.
	gx,gy = np.meshgrid(x,x,indexing = 'ij')

	bad = np.all(np.isnan(punches),axis=(1,2))
	punches[bad] = 0
	sx = np.nanmean(punches*gx[None,:,:],axis=(1,2))
	sy = np.nanmean(punches*gy[None,:,:],axis=(1,2))
	sx[np.abs(sx)>max_shift] = 0.
	sy[np.abs(sy)>max_shift] = 0.

	out = spots.copy()
	out[:,0] += sx
	out[:,1] += sy
	return out

def locate_good_localmax(img,localmax_region,acf_cutoff):
	dl = localmax_region
	localmaxes = minmax.local_max_mask(img[None,:,:],0,dl,dl)[0]
	spots = np.nonzero(localmaxes) ## [2,N]
	spots = (np.array(spots).T).astype('double') ## [N,2]
	intensities = img[localmaxes].flatten()
	keep = intensities >= acf_cutoff
	spots = spots[keep]
	return spots

def transform_back_spots(spots, transforms, ref_ind=0):
	## Put colors i spots back into color i space
	ncolors = transforms.shape[0]
	
	new_spots = [spotsi.copy() for spotsi in spots]
	
	for i in range(ncolors):
		if i == ref_ind:
			continue
		theta = transforms[ref_ind,i]
		order = alignment.coefficients_order(theta)
		a,b = alignment.coefficients_split(theta)
		sxi,syi = alignment.polynomial_transform_many(spots[i][:,0].copy(),spots[i][:,1].copy(),a,b,order)
		new_spots[i] = np.array((sxi,syi)).T
	return new_spots

def prep_images(img, transforms, smooth):
	## img --> analysis.img
	## transforms --> analysis.transforms
	
	ncolors = img.shape[0]
	imgs = [img[0],]
	for i in range(1,ncolors):
		revtheta = transforms[0,i] ## nb it's meant to be reverse transform, b/c you need to look up original pixel locations
		a,b = alignment.coefficients_split(revtheta)
		imgs.append(alignment.rev_interpolate_polynomial(img[i],a,b))
	if smooth > 0 :
		imgs = [gaussian_filter(img,smooth) for img in imgs]
	return imgs