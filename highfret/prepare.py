import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import numpy as np
import numba as nb
import os
import re

import tifffile

from .fast_median import median_scmos, med_huang_floatimg

try:
	import zarr
	def load(fname,first=None,last=None):
		print('loading')
		store = tifffile.imread(fname, aszarr=True)
		z = zarr.open(store, mode='r')
		if first is None:
			first = 0
		if last is None:
			last = z.shape[0]
		z = np.array(z[first:last],dtype='uint16')
		return z
except:
	def load(fn):
		return tifffile.imread(fn)

@nb.njit(cache=True)
def apply_calibration(data,cal):
	if data.shape[1] != cal.shape[1] or data.shape[2] != cal.shape[2]:
		raise Exception('Calibration is for a movie of a different size. Were you binning?')
	g,o,v = cal
	for t in range(data.shape[0]):
		for i in range(data.shape[1]):
			for j in range(data.shape[2]):
				dtemp = int((float(data[t,i,j])-o[i,j])/g[i,j])
				if dtemp < 0:
					dtemp = 0
				if dtemp > 2**16-1:
					dtemp = 2**16-1
				data[t,i,j] = dtemp
	# return data


def acf(movie,median_n=21):
	q = _acf(movie) ## this should be a double so don't use median_scmos
	# q = _acf_alt(movie) ## this should be a double so don't use median_scmos
	# from scipy.ndimage import median_filter 
	# bg = median_filter(q,median_n)
	bg = med_huang_floatimg(q,median_n)
	return q-bg

@nb.njit(cache=True)
# @nb.njit(nogil=True,parallel=True,fastmath=True)
def _acf_alt(movie):
	'''
	can take anything including unint
	outputs double
	'''
	nt,ni,nj = movie.shape
	img0 = np.zeros((ni,nj),dtype='double')
	img1 = np.zeros((ni,nj),dtype='double')
	med1 = float(np.median(movie[0]))
	med2 = float(med1)

	for t in range(1,nt):
		med2 = float(np.median(movie[t]))
		if np.isnan(med2):
			raise Exception('wtf')
		for i in range(ni):
			for j in range(nj):
				img1[i,j] += (float(movie[t,i,j])-med2)*(float(movie[t-1,i,j])-med1)
				img0[i,j] += (float(movie[t,i,j])-med2)**2.
		med1 = med2
	img0 /= float(nt-1)
	img1 /= float(nt-1)
	img1 /= img0
	return img1

# @nb.njit(cache=True)
# # @nb.njit(nogil=True,parallel=True,fastmath=True)
# def _acf(movie):
# 	'''
# 	can take anything including unint
# 	outputs double
# 	'''
# 	nt,ni,nj = movie.shape
# 	img = np.zeros((ni,nj),dtype=nb.double)
# 	mean = np.zeros((ni,nj),dtype=nb.double)
# 	mean2 = np.zeros((ni,nj),dtype=nb.double)
# 	for i in range(ni):
# 		for j in range(nj):
# 			for t in range(nt):
# 				mean[i,j] += float(movie[t,i,j])
# 				mean2[i,j] += float(movie[t,i,j])**2.
# 			mean[i,j] /= nt
# 			mean2[i,j] /= nt
# 	for i in range(ni):
# 		for j in range(nj):
# 			for t in range(1,nt):
# 				img[i,j] += (float(movie[t,i,j])-mean[i,j])*(float(movie[t-1,i,j])-mean[i,j])
# 			img[i,j] /= (nt-1)
# 			# img[i,j] /= mean2[i,j]
# 			if (mean2[i,j]-mean[i,j]**2.) == 0:
# 				img[i,j] = 0.
# 			else:
# 				img[i,j] /= (mean2[i,j]-mean[i,j]**2.)
# 	return img

@nb.njit(cache=True)
# @nb.njit(nogil=True,parallel=True,fastmath=True)
def _acf(movie):
	'''
	can take anything including unint
	outputs double
	'''
	tau = 5
	nt,ni,nj = movie.shape
	img = np.zeros((ni,nj),dtype=nb.double)
	mean = np.zeros((ni,nj),dtype=nb.double)
	mean2 = np.zeros((ni,nj),dtype=nb.double)
	for i in range(ni):
		for j in range(nj):
			for t in range(nt):
				mean[i,j] += float(movie[t,i,j])
				mean2[i,j] += float(movie[t,i,j])**2.
			mean[i,j] /= nt
			mean2[i,j] /= nt
	for i in range(ni):
		for j in range(nj):
			for t in range(tau,nt):
				img[i,j] += (float(movie[t,i,j])-mean[i,j])*(float(movie[t-tau,i,j])-mean[i,j])
			img[i,j] /= (nt-tau)
			# img[i,j] /= mean2[i,j]
			if (mean2[i,j]-mean[i,j]**2.) == 0:
				img[i,j] = 0.
			else:
				img[i,j] /= (mean2[i,j]-mean[i,j]**2.)
	return img

def bin2x(d):
	nt,nx,ny = d.shape
	mx = int(nx//2)
	my = int(ny//2)
	out = d[:,::2,::2] + d[:,1::2,::2] + d[:,::2,1::2] + d[:,1::2,1::2]
	if mx % 2 == 1:
		out = out[:,:-1]
	if my % 2 == 1:
		out = out[:,:,:-1]
	return out


def split_lr(d):
	if d.ndim == 2:
		half = d.shape[1]//2
		d1 = d[:,:half]
		d2 = d[:,half:2*half]
	elif d.ndim == 3:
		half = d.shape[2]//2
		d1 = d[:,:,:half]
		d2 = d[:,:,half:2*half]
	else:
		logger.debug("Aborted split: image dimensions not 2 or 3")
		raise Exception("Aborted split: image dimensions not 2 or 3")
	return d1,d2

def split_tb(d):
	if d.ndim == 2:
		half = d.shape[0]//2
		d1 = d[:half]
		d2 = d[half:2*half]
	elif d.ndim == 3:
		half = d.shape[1]//2
		d1 = d[:,:half]
		d2 = d[:,half:2*half]
	else:
		logger.debug("Aborted split: image dimensions not 2 or 3")
		raise Exception("Aborted split: image dimensions not 2 or 3")
	return d1,d2

def split_quad(d):
	if d.ndim == 2:
		d1 = d[:d.shape[0]//2,:d.shape[1]//2]
		d2 = d[:d.shape[0]//2,d.shape[1]//2:]
		d3 = d[d.shape[0]//2:,:d.shape[1]//2]
		d4 = d[d.shape[0]//2:,d.shape[1]//2:]
	elif d.ndim == 3:
		d1 = d[:,:d.shape[1]//2,:d.shape[2]//2]
		d2 = d[:,:d.shape[1]//2,d.shape[2]//2:]
		d3 = d[:,d.shape[1]//2:,:d.shape[2]//2]
		d4 = d[:,d.shape[1]//2:,d.shape[2]//2:]
	else:
		logger.debug("Aborted split: image dimensions not 2 or 3")
		raise Exception("Aborted split: image dimensions not 2 or 3")
	return d1,d2,d3,d4


def get_out_dir(fn_data):
	print(fn_data)
	filename = re.sub(r'\s+', '', fn_data)
	print(filename,os.path.exists(filename))
	if os.path.exists(filename):

		## pull out file name from path
		fn_path,fn_data = os.path.split(filename)

		## pull off extension
		if fn_data.endswith('.ome.tif'):
			fn_base = fn_data[:-8]
		else:
			fn_base = os.path.splitext(fn_data)[0]

		fn_out_dir = os.path.join(fn_path,'highfret_%s'%(fn_base))

		dirs = [
			fn_out_dir,
			os.path.join(fn_out_dir,'temp'),
			os.path.join(fn_out_dir,'aligner'),
			os.path.join(fn_out_dir,'spotfinder'),
			os.path.join(fn_out_dir,'extracter')
		]

		for dir in dirs:
			if not os.path.exists(dir):
				os.mkdir(dir)
		return dirs
	
	else:
		raise Exception('File does not exist')
	
def dump_job(fname,description='',job={}):
	with open(fname,'w') as f:
		f.write('%s\n'%(description))
		for key in job.keys():
			f.write('%s: %s\n'%(key,job[key]))

def find_tif_files(root_folder):
	tif_files = []
	for dirpath, dirnames, filenames in os.walk(root_folder):
		for filename in filenames:
			if filename.endswith('.ome.tif'):
				tif_files.append(os.path.join(dirpath, filename))
	return tif_files

