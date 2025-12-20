import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import time
import tifffile
import zarr
from tqdm import tqdm
import numpy as np
import numba as nb

from .fast_median import  med_huang_floatimg,median_scmos

# def load(fn):
# 	return tifffile.imread(fn)

def load(fn, bin:int=1, verbose:bool=True):
	assert bin > 0

	with tifffile.imread(fn, aszarr=True) as store:
		z = zarr.open(store, mode='r')
		nt,nx,ny = z.shape

		data = np.zeros((nt,nx//bin,ny//bin),dtype='uint16')

		tloop = tqdm(range(nt),desc='Loading Movie') if verbose else range(nt)
		for t in tloop:
			temp = np.zeros((nx//bin, ny//bin), dtype='int')
			for i in range(bin):
				for j in range(bin):
					temp += z[t, i:nx-nx%bin:bin, j:ny-ny%bin:bin]
			temp[temp>65535] = 65535
			temp[temp<0] = 0
			data[t] = temp.astype('uint16')
	return data

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


def acf(movie):
	q = _acf(movie) ## this should be a double so don't use median_scmos

	return q

# @nb.njit(cache=True)
@nb.njit(nogil=True,parallel=True,fastmath=True)
def _acf(movie):
	'''
	can take anything including unint
	outputs double
	'''
	tau = 1
	nt,ni,nj = movie.shape
	img = np.zeros((ni,nj),dtype=nb.double)
	mean = np.zeros((ni,nj),dtype=nb.double)
	mean2 = np.zeros((ni,nj),dtype=nb.double)
	for i in nb.prange(ni):
		for j in range(nj):
			for t in range(nt):
				mean[i,j] += float(movie[t,i,j])
				mean2[i,j] += float(movie[t,i,j])**2.
			mean[i,j] /= nt
			mean2[i,j] /= nt
	for i in nb.prange(ni):
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

class ColorArray:
	def __init__(self, channels):
		self._ch = tuple(channels)
		self.ndim = self._ch[0].ndim + 1
		self.shape = (len(self._ch),) + self._ch[0].shape
		self.dtype = self._ch[0].dtype

	def __array__(self): ## Explicit conversion -- makes a real copy of the array
		return np.stack(self._ch, axis=0)

	def __getitem__(self, idx):
		if not isinstance(idx, tuple):
			idx = (idx,)
		idx = idx + (slice(None),) * (self.ndim - len(idx))

		c_idx, rest = idx[0], idx[1:]

		# color selection
		if isinstance(c_idx, int):
			return self._ch[c_idx][rest]

		if isinstance(c_idx, slice):
			new_ch = tuple(ch[rest] for ch in self._ch[c_idx])
			return ColorArray(new_ch)

	def __setitem__(self, idx, value):
		if not isinstance(idx, tuple):
			idx = (idx,)
		idx = idx + (slice(None),) * (self.ndim - len(idx))

		c_idx, rest = idx[0], idx[1:]

		if isinstance(c_idx, int):
			self._ch[c_idx][rest] = value
			return

		if isinstance(c_idx, slice):
			channels = self._ch[c_idx]
			if isinstance(value, ColorArray):
				for ch, v in zip(channels, value._ch):
					ch[rest] = v
			else:
				for ch in channels:
					ch[rest] = value
			return

	def __repr__(self):
		return f"ColorArray(shape={self.shape}, dtype={self.dtype})"


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
		raise Exception("Aborted split: image dimensions not 2 or 3")
	return ColorArray((d1,d2))

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
		raise Exception("Aborted split: image dimensions not 2 or 3")
	return ColorArray((d1,d2))

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
		raise Exception("Aborted split: image dimensions not 2 or 3")
	return ColorArray((d1,d2,d3,d4))