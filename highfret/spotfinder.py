import tifffile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .containers import general_analysis_class
from .support.spotfinder import compare_close_spots,remove_close_spots,find_outofframe,refine_simple,locate_good_localmax,transform_back_spots,prep_images
from .support.modelselect_alignment import nps2rgb

def find_spots(analysis: general_analysis_class, smooth, localmax_region, acf_cutoff, refine, matched_spots, which):
	ncolors = analysis.img.shape[0]
	
	analysis.log += f'Spotfinding:\n'
	#### STEP 1
	## Transform all into color space 0
	analysis.log += f'\tTransforming all images into color space 0\n'
	imgs = prep_images(analysis.img, analysis.transforms, smooth)

	#### Find color i spots in color 0 coordinates
	separate_spots = [locate_good_localmax(img,localmax_region,acf_cutoff) for img in imgs]
	analysis.log += f'\tLocated Spots -- {",".join([f"Color {i}: {separate_spots[i].shape[0]}" for i in range(len(separate_spots))])}\n'

	if refine:
		separate_spots = [refine_simple(imgs[i],separate_spots[i]) for i in range(ncolors)]
		analysis.log += f'\tRefined reduced spots\n'
	
	#### STEP 2
	reduced_spots = [spotsi.copy() for spotsi in separate_spots]	

	## Combine the spots to just get those that match
	if matched_spots:
		## First collect matches into color 0
		for i in range(1,ncolors):
			reduced_spots[0] = compare_close_spots(reduced_spots[0],reduced_spots[i],1.5)[0] ## only keep spots with a match. 1.5 ~ sqrt[2]
		## Propagated matched into all other colors
		for i in range(1,ncolors):
			reduced_spots[i] = compare_close_spots(reduced_spots[0],reduced_spots[i],1.5)[1] ## only keep spots with a match. 1.5 ~ sqrt[2]
		analysis.log += f'\tMatched Spots -- {",".join([f"Color {i}: {reduced_spots[i].shape[0]}" for i in range(len(reduced_spots))])}\n'

	## Combine & remove duplicates
	if which == 'all':
		good_spots = np.concatenate(reduced_spots,axis=0) ## all spots in green coordinates
	elif which.lower().startswith('only '):
		good_spots = reduced_spots[int(which[-1])]
	else:
		raise Exception('IDK which spots you want')
	analysis.log += (f'\tSpots from: {which}')
	
	good_spots = remove_close_spots(good_spots,1.5) ## only keep spots apart by 1.5 ~ sqrt[2]
	analysis.log += f'\tReduced spots: {good_spots.shape[0]}\n'

	#### STEP 3
	## Put colors i spots back from color 0 space into color i space
	reduced_spots = [good_spots,]*ncolors
	separate_spots = transform_back_spots(separate_spots,analysis.transforms)
	reduced_spots = transform_back_spots(reduced_spots,analysis.transforms)
	analysis.log += f'\tTransformed spots back to orginal color space\n'
	
	## Remove spots that're out of frame...
	keep = np.isfinite(good_spots[:,0])
	for i in range(ncolors):
		separate_spots[i] = separate_spots[i][find_outofframe(imgs[i],separate_spots[i])]
		keep = np.bitwise_and(keep,find_outofframe(imgs[i],reduced_spots[i]))
	for i in range(ncolors):
		reduced_spots[i] = reduced_spots[i][keep]
	analysis.log += f'\tRemoved out of frame separate spots -- {",".join([f"Color {i}: {separate_spots[i].shape[0]}" for i in range(len(separate_spots))])}\n'
	analysis.log += f'\tRemoved out of frame reduced  spots -- {",".join([f"Color {i}: {reduced_spots[i].shape[0]}" for i in range(len(reduced_spots))])}\n'
	
	if refine:
		reduced_spots = [refine_simple(imgs[i],reduced_spots[i]) for i in range(ncolors)]
		analysis.log += f'\tRefined reduced spots\n'

	return separate_spots, np.array(reduced_spots)

def make_aligned_tif(analysis: general_analysis_class, output_directory: Path, smooth: float=0.5):
	imgs = prep_images(analysis.img, analysis.transforms, smooth)
	ncolors = len(imgs)
	nx,ny = imgs[0].shape

	### make full transformed image
	full = np.zeros((nx,ny*ncolors))
	for i in range(ncolors):
		full[:,i*ny:(i+1)*ny] = imgs[i]
	## it's an acf image so clip on [0:1]
	full[full<0] = 0
	full[full>1] = 1
	full *= 2**16
	full = full.astype('uint16')

	fn_out = Path(output_directory) / 'composite_aligned.tif'
	tifffile.imwrite(fn_out, full)
	analysis.log += f'Wrote composite aligned .tif image to: {fn_out}\n'

def render_overlay(analysis: general_analysis_class, smooth: float):
	#### to check the alignment of ACF images
	imgs = prep_images(analysis.img, analysis.transforms, smooth)
	ncolors = len(imgs)
	assert ncolors <= 3

	img = nps2rgb(imgs[0],imgs[0])
	if ncolors >= 2:
		img[:,:,0] = nps2rgb(imgs[1],imgs[1])[:,:,0] ## make overlay image to copy
	if ncolors >= 3:
		img[:,:,2] = nps2rgb(imgs[2],imgs[2])[:,:,2] ## make overlay image to copy

	## clip on [0:1] b/c it's an ACF image
	img[img<0] = 0
	img[img>1] = 1

	fig,ax=plt.subplots(1)
	ax.imshow(img,interpolation='nearest',vmin=0,vmax=1) ## vmin/vmax b/c acf in [0,1] ish
	ax.set_axis_off()
	ax.set_title('Overlay')
	return fig,ax

def render_found_maxes(analysis: general_analysis_class, spots):
	#### plots [np.ndarrays...] not analysis.spots
	#### to make sure all found spots in original coordinates are good
	ncolors = analysis.img.shape[0]
	assert len(spots) == ncolors

	color_cycle = ['tab:green','tab:red',]
	fig,ax = plt.subplots(1,ncolors,sharex=True,sharey=True)	
	for i in range(ncolors):
		ax[i].imshow(analysis.img[i],cmap='Greys_r',vmin=0,vmax=1,interpolation='nearest')
		if i < len(color_cycle):
			ax[i].scatter(spots[i][:,1],spots[i][:,0],alpha=.5,facecolor='none',edgecolor=color_cycle[i],)
		else:
			ax[i].scatter(spots[i][:,1],spots[i][:,0],alpha=.5,facecolor='none')
		ax[i].set_title(f'Color {i}')
	[aa.set_axis_off() for aa in ax]
	return fig,ax

def render_final_spots(analysis: general_analysis_class, spots):
	#### plots [np.ndarrays...] not analysis.spots
	#### to check that final spots are good
	ncolors = analysis.img.shape[0]
	
	fig,ax = plt.subplots(1,ncolors,sharex=True,sharey=True)
	for i in range(ncolors):
		ax[i].imshow(analysis.img[i],cmap='Greys_r',vmin=0,vmax=1,interpolation='nearest')
		ax[i].scatter(spots[i][:,1],spots[i][:,0],edgecolor='tab:orange',alpha=.5,facecolor='none')
		ax[i].set_title(f'Color {i+1}')
	[aa.set_axis_off() for aa in ax]
	
	return fig,ax

def run_spotfind(
		analysis: general_analysis_class,
		acf_cutoff: float=0.15,
		matched_spots: bool=True,
		smooth: float=0.5,
		which: str="all",
		localmax_region: int=1,
		refine: bool=True,
	):

	separate_spots,reduced_spots = find_spots(analysis, smooth, localmax_region, acf_cutoff, refine, matched_spots, which)
	analysis.spots = reduced_spots

	fig1,ax1 = render_found_maxes(analysis,separate_spots)
	fig1.set_figheight(8.)
	fig1.set_figwidth(8.)
	
	fig2,ax2 = render_final_spots(analysis,reduced_spots)
	fig2.set_figheight(8.)
	fig2.set_figwidth(8.)

	return fig1,fig2