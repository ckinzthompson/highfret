import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import os
import time
import tifffile
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from . import prepare,minmax,alignment,punch


def default_flags():
	df = {
		'fn_data':None,
		'fn_align':None,
		'fn_cal':None,

		'split':'L/R',
		'acf_cutoff':0.2,
		'matched_spots':True,
		'acf_start_g':0,
		'acf_end_g':0,
		'acf_start_r':0,
		'acf_end_r':0,
		'smooth':0.6,
		'which':'Both',
		'localmax_region':1,
		'refine':True,
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

@nb.njit
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

def prepare_data(fn_data,fn_cal,flags):
	print('Loading')
	data = prepare.load(fn_data)

	if fn_cal is None:
		calibration = np.zeros((3,data.shape[1],data.shape[2])) ## _,o,v
		calibration[0] += 1. ## g,_,_
	else:
		calibration = np.load(fn_cal) ## g,o,v
	
	print('Calibrating')
	prepare.apply_calibration(data,calibration) ## preparation is in place
	
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
	imgg = prepare.acf(dg[flags['acf_start_g']:end_g])
	imgr = prepare.acf(dr[flags['acf_start_r']:end_r])

	dirs = prepare.get_out_dir(fn_data)
	out_dir_temp = dirs[1]
	out_dir = dirs[3]

	print('Prepared Shapes: %s, %s'%(str(imgg.shape),str(imgr.shape)))
	np.save(os.path.join(out_dir_temp,'prep_imgg.npy'),imgg)
	np.save(os.path.join(out_dir_temp,'prep_imgr.npy'),imgr)

	with open(os.path.join(out_dir,'details_spotfinder.txt'),'w') as f:
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
	dirs = prepare.get_out_dir(fn_data)
	out_dir_temp = dirs[1]
	out_dir = dirs[3]

	if not os.path.exists(os.path.join(out_dir_temp,'prep_imgg.npy')) or not os.path.exists(os.path.join(out_dir_temp,'prep_imgr.npy')):
		raise Exception('Please run prepare_data first')
		
	img1 = np.load(os.path.join(out_dir_temp,'prep_imgg.npy'))
	img2 = np.load(os.path.join(out_dir_temp,'prep_imgr.npy'))

	return img1,img2

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

def prep_imgs(imgg,imgr,theta,flags,align=True):
	if flags['smooth'] > 0:
		imgg = gaussian_filter(imgg,flags['smooth'])
		imgr = gaussian_filter(imgr,flags['smooth'])
	
	if align:
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

	if flags['refine']:
		g_spots_g = refine_simple(imgg,g_spots_g)
		g_spots_r = refine_simple(imgr,g_spots_r)

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
	
	keep = find_outofframe(imgr,r_spots)
	g_spots = g_spots[keep]
	r_spots = r_spots[keep]

	if flags['refine']:
		imgg,imgr = get_prepared_data(fn_data)
		imgg,imgr = prep_imgs(imgg,imgr,theta,flags,align=False)
		g_spots = refine_simple(imgg,g_spots)
		r_spots = refine_simple(imgr,r_spots)

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

	dirs = prepare.get_out_dir(fn_data)
	out_dir_temp = dirs[1]
	out_dir = dirs[3]

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

def save_spots(fn_data,g_spots_g,g_spots_r,r_spots_r,g_spots,r_spots):
	dirs = prepare.get_out_dir(fn_data)
	out_dir_temp = dirs[1]
	out_dir = dirs[3]

	np.save(os.path.join(out_dir_temp,'%s.npy'%('g_spots_g')),g_spots_g)
	np.save(os.path.join(out_dir_temp,'%s.npy'%('g_spots_r')),g_spots_r)
	np.save(os.path.join(out_dir_temp,'%s.npy'%('r_spots_r')),r_spots_r)
	np.save(os.path.join(out_dir_temp,'%s.npy'%('g_spots')),g_spots)
	np.save(os.path.join(out_dir_temp,'%s.npy'%('r_spots')),r_spots)


def run_job_prepare(job):
	fn_data = job['fn_data']
	fn_align = job['fn_align']
	fn_cal = job['fn_cal']
	flags = job

	dirs = prepare.get_out_dir(fn_data)
	dir_spotfinder = dirs[3]

	prepare_data(fn_data,fn_cal,flags)
	make_composite_aligned(fn_data,fn_align,flags)

	fig,ax = render_overlay(fn_data,fn_align,flags)
	fig.set_figheight(8.)
	fig.set_figwidth(8.)
	[plt.savefig(os.path.join(dir_spotfinder,'overlay.%s'%(ext))) for ext in ['png','pdf']]

	prepare.dump_job(os.path.join(dir_spotfinder,'job_prepare.txt'),'Job Name: Prepare for Spotfinding',job)

	return fig

def run_job_spotfind(job):
	fn_data = job['fn_data']
	fn_align = job['fn_align']
	flags = job

	dirs = prepare.get_out_dir(fn_data)
	dir_spotfinder = dirs[3]

	g_spots_g,g_spots_r,r_spots_r,g_spots,r_spots = find_spots(fn_data,fn_align,flags)
	save_spots(fn_data,g_spots_g,g_spots_r,r_spots_r,g_spots,r_spots)
	
	fig1,ax1 = render_found_maxes(fn_data,fn_align,g_spots_g,g_spots_r,flags)
	fig1.set_figheight(8.)
	fig1.set_figwidth(8.)
	[plt.savefig(os.path.join(dir_spotfinder,'spots_all.%s'%(ext))) for ext in ['png','pdf']]

	fig2,ax2 = render_final_spots(fn_data,g_spots,r_spots,flags)
	fig2.set_figheight(8.)
	fig2.set_figwidth(8.)
	[plt.savefig(os.path.join(dir_spotfinder,'spots_final.%s'%(ext))) for ext in ['png','pdf']]

	prepare.dump_job(os.path.join(dir_spotfinder,'job_spotfind.txt'),'Job Name: Spotfinding',job)

	return fig1,fig2