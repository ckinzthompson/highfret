import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import os
import re
import time
import tifffile
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from . import prepare,minmax,alignment


def default_flags():
	df = {
		'split':'L/R',
		'acf_cutoff':0.2,
		'matched_spots':False,
		'acf_start_g':0,
		'acf_end_g':0,
		'acf_start_r':0,
		'acf_end_r':0,
		'smooth':0.6,
		'which':'Both',
		'localmax_region':1,
	}
	return df

@nb.njit
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

@nb.njit
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

		fn_out_dir = os.path.join(fn_path,'spotfinder_results_%s'%(fn_base))
		return fn_out_dir
	else:
		raise Exception('File does not exist')

def prepare_data(fn_data,fn_cal,flags):
	print('Loading')
	data = prepare.load(fn_data)

	if fn_cal is None:
		calibration = np.zeros((3,data.shape[1],data.shape[2])) ## _,o,v
		calibration[0] += 1. ## g,_,_
	else:
		calibration = np.load(fn_cal) ## g,o,v
	
	print('Calibrating')
	data = prepare.apply_calibration(data,calibration)
	
	print('Splitting')
	if flags['split'] == 'L/R':
		dg,dr = prepare.split_lr(data)
	elif flags['split'] == 'T/B':
		dg,dr = prepare.split_tb(data)

	
	end_g = dg.shape[0] if flags['acf_end_g'] == 0 else flags['acf_end_g']
	end_r = dr.shape[0] if flags['acf_end_r'] == 0 else flags['acf_end_r']
	print('ACF (Green) Start/End: %d/%d'%(flags['acf_start_g'],end_g))
	print('ACF (Red)   Start/End: %d/%d'%(flags['acf_start_g'],end_r))
	print('ACFing')
	imgg = prepare.acf1(dg[flags['acf_start_g']:end_g])
	imgr = prepare.acf1(dr[flags['acf_start_r']:end_r])

	out_dir = get_out_dir(fn_data)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	print('Prepared Shapes: %s, %s'%(str(imgg.shape),str(imgr.shape)))
	np.save(os.path.join(out_dir,'prep_temp_imgg.npy'),imgg)
	np.save(os.path.join(out_dir,'prep_temp_imgr.npy'),imgr)

	with open(os.path.join(out_dir,'prep_details.txt'),'w') as f:
		out = "Aligner - %s\n=====================\n"%(time.ctime())
		out += '%s \n'%(fn_data)
		out += '%s \n=====================\n'%(str(data.shape))
		out += 'Avg. Cal: Gain  : %.2f'%(calibration[0].mean())
		out += 'Avg. Cal: Offset: %.2f'%(calibration[1].mean())
		if flags['split'] == 'L/R':
			out += 'Split: Left/Right\n'
		else:
			out += 'Split: Top/Bottom\n'
		out += 'ACF (Green) Start/End: %d/%d'%(flags['acf_start_g'],end_g)
		out += 'ACF (Red)   Start/End: %d/%d'%(flags['acf_start_r'],end_r)
		f.write(out)
	print('Preparation Done')

def get_prepared_data(fn_data):
	#### Load prepared image
	out_dir = get_out_dir(fn_data)

	if not os.path.exists(os.path.join(out_dir,'prep_temp_imgg.npy')) or not os.path.exists(os.path.join(out_dir,'prep_temp_imgr.npy')):
		raise Exception('Please run prepare_data first')
		
	img1 = np.load(os.path.join(out_dir,'prep_temp_imgg.npy'))
	img2 = np.load(os.path.join(out_dir,'prep_temp_imgr.npy'))

	return img1,img2

def prep_imgs(imgg,imgr,theta,flags):
	if flags['smooth'] > 0:
		imgg = gaussian_filter(imgg,flags['smooth'])
		imgr = gaussian_filter(imgr,flags['smooth'])
	
	######## N.B. invert is not reliable
	#### Do everything in green space
	## theta is R to G, but interp happens backwards (ie for theta, it's G to R)
	## therefore inv_theta (G to R) gives ability to transform R to G
	inv_theta = alignment.invert_transform(theta,shape=imgg.shape,nl=40) 
	a,b = alignment.coefficients_split(inv_theta)
	imgr = alignment.rev_interpolate_polynomial(imgr,a,b) ## img2 is R in green space 
	return imgg,imgr

def locate_good_localmax(img,flags):
	dl = flags['localmax_region']
	localmaxes = minmax.local_max_mask(img[None,:,:],0,dl,dl)[0]
	spots = np.nonzero(localmaxes) ## [2,N]
	spots = (np.array(spots).T).astype('double') ## [N,2]
	intensities = img[localmaxes].flatten()
	keep = intensities >= flags['acf_cutoff']
	spots = spots[keep]
	return spots

def find_spots(fn_data,fn_align,flags):
	#### Nomenclature:
	#### <color space>_spots_<data space> -- ie, g_spots_r are red spots in the green coordinate
	print('Finding Spots')
	theta = np.load(fn_align)
	imgg,imgr = get_prepared_data(fn_data)
	imgg,imgr = prep_imgs(imgg,imgr,theta,flags)

	#### Find spots
	g_spots_g = locate_good_localmax(imgg,flags) ## green spots in green coordinates
	g_spots_r = locate_good_localmax(imgr,flags) ## red spots in green coordinates

	#### Compile spots
	print('Compiling Spots')
	## Combine the spots to just get those that match
	if flags['matched_spots']: 
		print('Only Matched Spots')
		spots_gg,spots_rr = compare_close_spots(g_spots_g,g_spots_r,1.5) ## only keep spots with a match. 1.5 > sqrt[2]
	else:
		print('Non-matched Spots')
		spots_gg = g_spots_g.copy()
		spots_rr = g_spots_r.copy()

	## Combine & remove duplicates
	if flags['which'] == 'Both':
		g_spots = np.concatenate([spots_gg,spots_rr],axis=0) ## all spots in green coordinates
	elif flags['which'] == 'Only Green':
		g_spots = spots_gg.copy()
	elif flags['which'] == 'Only Red':
		g_spots = spots_rr.copy()
	else:
		raise Exception('IDK which spots you want')
	print('Spots from %s'%flags['which'])
	g_spots = remove_close_spots(g_spots,1.5) ## combine mached spots to their average position

	## Put red spots back into red space
	inv_theta = alignment.invert_transform(theta,shape=imgg.shape,nl=40)
	# inv_theta = np.array([0.,1.,0.,0.,0.,1.]) ## for debugging
	order = alignment.coefficients_order(inv_theta) ## prep transform
	a,b = alignment.coefficients_split(inv_theta) ## prep transform
	sxr,syr = alignment.polynomial_transform_many(g_spots_r[:,0].copy(),g_spots_r[:,1].copy(),a,b,order)
	r_spots_r = np.array((sxr,syr)).T ## red spots in green coordinates

	sxr,syr = alignment.polynomial_transform_many(g_spots[:,0].copy(),g_spots[:,1].copy(),a,b,order)
	r_spots = np.array((sxr,syr)).T ## all spots in red coordinates

	return g_spots_g,g_spots_r,r_spots_r,g_spots,r_spots

def make_composite_aligned(fn_data,fn_align,flags):
	theta = np.load(fn_align)
	imgg,imgr = get_prepared_data(fn_data)
	imgg,imgr = prep_imgs(imgg,imgr,theta,flags)

	### make full transformed image
	full = np.zeros((imgg.shape[0],imgr.shape[1]*2))
	full[:,:imgg.shape[1]] = imgg
	full[:,imgg.shape[1]:] = imgr
	full[full<0] = 0
	full[full>1] = 1
	full *= 2**16
	full = full.astype('uint16')

	out_dir = get_out_dir(fn_data)
	fn_out = os.path.join(out_dir,'composite_aligned.tif')
	tifffile.imwrite(fn_out, full)

def render_overlay(fn_data,fn_align,flags):
	#### to check the alignment of ACF images
	theta = np.load(fn_align)
	imgg,imgr = get_prepared_data(fn_data)
	imgg,imgr = prep_imgs(imgg,imgr,theta,flags)

	img = alignment.nps2rgb(imgg,imgr) ## make overlay image
	img[img<0] = 0
	img[img>1] = 1

	fig,ax=plt.subplots(1)
	ax.imshow(img,interpolation='nearest',vmin=0,vmax=1) ## vmin/vmax b/c acf in [0,1] ish
	ax.set_axis_off()
	ax.set_title('Overlay')
	return fig,ax

def render_found_maxes(fn_data,fn_align,g_spots_g,g_spots_r,flags):
	#### to make sure all found spots in original coordinates are good
	imgg,imgr = get_prepared_data(fn_data)
	theta = np.load(fn_align)
	imgg,imgr = prep_imgs(imgg,imgr,theta,flags)

	fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
	
	ax[0].imshow(imgg,cmap='Greys_r',vmin=0,vmax=1,interpolation='nearest')
	ax[1].imshow(imgr,cmap='Greys_r',vmin=0,vmax=1,interpolation='nearest')
	ax[0].scatter(g_spots_g[:,1],g_spots_g[:,0],edgecolor='tab:green',alpha=.5,facecolor='none')
	ax[1].scatter(g_spots_r[:,1],g_spots_r[:,0],edgecolor='tab:red',alpha=.5,facecolor='none')
	
	[aa.set_axis_off() for aa in ax]
	ax[0].set_title('Green')
	ax[1].set_title('Red')
	return fig,ax

def render_final_spots(fn_data,g_spots,r_spots,flags):
	#### to check that final spots are good
	imgg,imgr = get_prepared_data(fn_data)
	
	fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
	ax[0].imshow(imgg,cmap='Greys_r',vmin=0,vmax=1,interpolation='nearest')
	ax[1].imshow(imgr,cmap='Greys_r',vmin=0,vmax=1,interpolation='nearest')

	ax[0].scatter(g_spots[:,1],g_spots[:,0],edgecolor='tab:orange',alpha=.5,facecolor='none')
	ax[1].scatter(r_spots[:,1],r_spots[:,0],edgecolor='tab:orange',alpha=.5,facecolor='none')

	[aa.set_axis_off() for aa in ax]
	ax[0].set_title('Green')
	ax[1].set_title('Red')
	return fig,ax

def save_spots(prefix,fn_data,g_spots_g,g_spots_r,r_spots_r,g_spots,r_spots):
	out_dir = get_out_dir(fn_data)
	np.save(os.path.join(out_dir,'%s_%s.npy'%(prefix,'g_spots_g')),g_spots_g)
	np.save(os.path.join(out_dir,'%s_%s.npy'%(prefix,'g_spots_r')),g_spots_r)
	np.save(os.path.join(out_dir,'%s_%s.npy'%(prefix,'r_spots_r')),r_spots_r)
	np.save(os.path.join(out_dir,'%s_%s.npy'%(prefix,'g_spots')),g_spots)
	np.save(os.path.join(out_dir,'%s_%s.npy'%(prefix,'r_spots')),r_spots)

