import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import numpy as np
import numba as nb

def spots_to_map(img,spots,n_clip=5):
	mask = np.zeros(img.shape,dtype='bool')[None,:,:]
	mask[0,spots[:,0],spots[:,1]] = True
	mask = clip_edges(mask,(0,n_clip,n_clip))
	return mask[0]

@nb.njit#(nogil=True,parallel=True,fastmath=True)
def local_max_mask(data,ni,nj,nk):

	if data.ndim == 2:
		raise Exception('Must be 3d... just reshape as data[None,:,:]')
		data = data.reshape((1,data.shape[0],data.shape[1]))
	nx, ny, nz = data.shape
	mask = np.ones((nx,ny,nz),dtype=nb.boolean)

	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				dd = data[i,j,k]
				mask[i,j,k] = False
				if i <= nx-ni and i >= ni and j <= ny-nj and j >= nj and k <= nz-nk and k >= nk:
					mask[i,j,k] = True
					for ii in range(max((0,i-ni)), min((i+ni+1,nx))):
						if mask[i,j,k] == False:
							break
						for jj in range(max((0,j-nj)), min((j+nj+1,ny))):
							if mask[i,j,k] == False:
								break
							for kk in range(max((0,k-nk)), min((k+nk+1,nz))):
								if data[ii,jj,kk] >= dd:
									if not (ii==i and jj==j and kk==k):
										mask[i,j,k] = False
										break
	return mask

@nb.njit#(nogil=True,parallel=True,fastmath=True)
def local_min_mask(data,ni,nj,nk):

	if data.ndim == 2:
		raise Exception('Must be 3d... just reshape as data[None,:,:]')
		data = data.reshape((1,data.shape[0],data.shape[1]))
	nx, ny, nz = data.shape
	mask = np.ones(data.shape,dtype=nb.boolean)

	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				dd = data[i,j,k]
				mask[i,j,k] = False
				if i <= nx-ni and i >= ni and j <= ny-nj and j >= nj and k <= nz-nk and k >= nk:
					mask[i,j,k] = True
					for ii in range(max((0,i-ni)), min((i+ni+1,nx))):
						if mask[i,j,k] == False:
							break
						for jj in range(max((0,j-nj)), min((j+nj+1,ny))):
							if mask[i,j,k] == False:
								break
							for kk in range(max((0,k-nk)), min((k+nk+1,nz))):
								if data[ii,jj,kk] <= dd:
									if not (ii==i and jj==j and kk==k):
										mask[i,j,k] = False
										break
	return mask

def _get_shape(data,footprint):
	## footprint is a tuple of dimensions in data
	shape = None
	if isinstance(footprint,int):
		shape = [footprint]*data.ndim
	elif isinstance(footprint,tuple) or isinstance(footprint,list):
		if len(footprint) == data.ndim:
			shape = footprint
	return shape

def clip_edges(data,footprint,value=0):
	'''
	minmap = clip_edges(minmap,(0,nclip,nclip))
	'''
	shape = _get_shape(data,footprint)
	if shape is None:
		logger.warning('Footprint is bad')
		return data

	for i in range(len(shape)):
		if shape[i] > 0:
			s = [slice(0,data.shape[i]) for i in range(len(shape))]
			s[i] = slice(0,shape[i])
			data[tuple(s)] = value
			s[i] = slice(data.shape[i]-shape[i],data.shape[i])
			data[tuple(s)] = value
	return data

def nearest_neighbors(mask,cutoff,keep=None):
	spots = np.array(np.nonzero(mask.sum(0))).T
	if not keep is None:
		spots = spots[keep]
	dist = np.sqrt((spots[:,None,0]-spots[None,:,0])**2. + (spots[:,None,1]-spots[None,:,1])**2.)
	keep = np.bitwise_and(dist<cutoff,dist > 0)
	nn_list = [np.nonzero(keep[i]) for i in range(dist.shape[0])]
	for i in range(len(nn_list)):
		if len(nn_list[i])>1:
			nn_list[i] = nn_list[i][1]
		else:
			nn_list[i] = nn_list[i][0] ## handy source of int zeros
	return nn_list
