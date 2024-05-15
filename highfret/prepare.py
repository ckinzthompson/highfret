import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import numpy as np
import numba as nb

import tifffile

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

@nb.njit
def apply_calibration(data,cal):
	g,o,v = cal
	for t in range(data.shape[0]):
		for i in range(data.shape[1]):
			for j in range(data.shape[2]):
				dtemp = int((float(data[t,i,j])-o[i,j])/g[i,j])
				if dtemp < 0:
					dtemp = 0
				data[t,i,j] = dtemp
	return data

@nb.njit
def acf1(movie):
	'''
	can take anything including unint
	outputs double
	'''
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
			for t in range(1,nt):
				img[i,j] += (float(movie[t,i,j])-mean[i,j])*(float(movie[t-1,i,j])-mean[i,j])
			img[i,j] /= (nt-1)
			# img[i,j] /= mean2[i,j]
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
