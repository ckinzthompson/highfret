import numpy as np
import numba as nb
from math import erf

## stolen from HARP: https://github.com/bayes-shape-calc/HARP

class gridclass(object):
	'''
	Convienience class to keep track of grided density data
	'''
	def __init__(self,origin,dxy,nxy,*args,**kwargs):
		self.origin = origin.astype('double')
		self.dxy = dxy.astype('double')
		self.nxy = nxy.astype('int64')

def x2i(grid,point):
	''' transfer xy to ij on grid '''
	return ((point-grid.origin)//grid.dxy).astype('int64')
def i2x(grid,point):
	''' transfer ij on grid to xy'''
	return (point*grid.dxy + grid.origin).astype('float64')

def subgrid_center(grid,centerxy,halfwidthxy):
	'''
	create a cubic grid centered around centerxy padded by halfwidthxy
	'''
	# if not xy_in_grid(grid,centerxy):
	# 	print('Centering: %s not in grid'%(str(centerxy)))
	ijmin = x2i(grid,centerxy-halfwidthxy)
	ijmax = x2i(grid,centerxy+halfwidthxy)+1
	ijmin[ijmin < 0] = 0
	upper = ijmax >= grid.nxy
	ijmax[upper] = grid.nxy[upper]
	newnxy = ijmax - ijmin
	neworigin = i2x(grid,ijmin)
	return gridclass(neworigin,grid.dxy,newnxy)

def subgrid_extract(grid,data,subgrid):
	'''
	extract density for subgrid from (grid,data) pair
	'''
	ijmin = x2i(grid,subgrid.origin)
	ijmax = ijmin+ subgrid.nxy
	ijmin[ijmin<0] = 0
	ijmax[ijmax<0] = 0
	return data[:,ijmin[0]:ijmax[0],ijmin[1]:ijmax[1]].copy()


@nb.njit(cache=True) ## don't parallelize this b/c otherwise race conditions
def render_model(origin,dxy,nxy,xy,weights,sigma,nsigma_cutoff,offset):
	'''
	sum up for several spots: integrate gaussian over voxels close to spot
	gives a decent speed up
	'''

	dmodel = np.zeros((nb.int64(nxy[0]),nb.int64(nxy[1])))

	## face or edge centered...
	offsethigh = 1. - offset
	offsetlow  = 0. - offset
	oxh = offsethigh*dxy[0]
	oxl = offsetlow*dxy[0]
	oyh = offsethigh*dxy[1]
	oyl = offsetlow*dxy[1]

	for atomi in range(xy.shape[0]):
		r2ss = np.sqrt(2.*sigma[atomi]*sigma[atomi])
		xyi = xy[atomi]
		xyimin = xyi - nsigma_cutoff*sigma[atomi]
		xyimax = xyi + nsigma_cutoff*sigma[atomi]

		ijimin = ((xyimin - origin)//dxy)//1
		ijimax = ((xyimax - origin)//dxy)//1

		for ii in range(max(0,int(ijimin[0])),min(int(ijimax[0])+1,dmodel.shape[0])):
				di = (ii*dxy[0] + origin[0]) - xyi[0] ## distances from mu
				for ji in range(max(0,int(ijimin[1])),min(int(ijimax[1])+1,dmodel.shape[1])):
						dj = (ji*dxy[1] + origin[1]) - xyi[1]
						c = (erf((di+oxh)/r2ss) - erf((di+oxl)/r2ss))
						c *= (erf((dj+oyh)/r2ss) - erf((dj+oyl)/r2ss))
						dmodel[ii,ji] += .25*c*weights[atomi]

	return dmodel

def density_point(grid,point,weight=1.,sigma=.7,nsigma=5,offset=0.5):
	'''
	single point calculation
	note: It seems like face centered is used -- 4f35 gives higher <b> of .47 to .29 for face vs edge.
	'''
	weights = np.array((weight,))
	sigmas = np.array((sigma,))
	return render_model(grid.origin,grid.dxy,grid.nxy,point[None,:],weights,sigmas,nsigma,offset)

def density_atoms(grid,xys,weights=None,sigma=.7,nsigma=5,offset=0.5):
	'''
	multiple point calculation
	'''
	natoms,_ = xys.shape
	if weights is None:
		weights = np.ones(natoms)
	if np.size(sigma)==1:
		sigmas = np.zeros((natoms))+sigma
	else:
		sigmas = sigma
	
	return render_model(grid.origin,grid.dxy,grid.nxy,xys,weights,sigmas,nsigma,offset)


@nb.njit(nogil=True,fastmath=True,parallel=True,cache=True)
def mle_all(subdata,subgrid_origin,subgrid_dxy,subgrid_nxy,spots,keep_neighbors,nmoli,NB0,sigma,nsigma,offset):
	nt,nsx,nsy = subdata.shape
	nmol,_ = spots.shape

	NBi = np.zeros((2,nt))

	## Figure out neighbor list
	mask = keep_neighbors[nmoli]
	nneighbors = np.sum(mask)
	neighbors = np.zeros((nneighbors),dtype='int')
	jj = 0
	for nmolj in range(nmol):
		if mask[nmolj]:
			neighbors[jj] = nmolj
			jj+=1
	
	## Make templates
	psis = np.zeros((nneighbors,subdata.shape[1],subdata.shape[2]))
	weights = np.ones((nneighbors))
	sigmas = np.zeros((nneighbors)) + sigma
	for nni in range(nneighbors):
		# print(subgrid_origin,subgrid_dxy,subgrid_nxy,spots[neighbors[nni],],weights,sigmas,nsigma,offset)
		spot = np.zeros((1,2)) ## it expects many spots
		spot[0] = spots[neighbors[nni]] 
		psis[nni] = render_model(subgrid_origin,subgrid_dxy,subgrid_nxy,spot,weights,sigmas,nsigma,offset)
		if neighbors[nni] == nmoli:
			sumpsi2 = np.sum(psis[nni]**2.)

	## Calculate N,B
	for t in nb.prange(nt):

		#### This is the median trick -- only use on bg. slows down but a little more robust for small areas
		sorter = np.zeros((nsx*nsy))
		NBi[1,t] = 0.
		for i in range(nsx):
			for j in range(nsy):
				numerator = subdata[t,i,j]
				for nni in range(nneighbors):
					nnn = neighbors[nni]
					numerator -= NB0[0,nnn,t]*psis[nni,i,j]
				sorter[i*nsy+j] = numerator
		NBi[1,t] = np.median(sorter)

		# #### This is the regular bg
		# NBi[1,t] = 0.
		# for i in range(nsx):
		# 	for j in range(nsy):
		# 		numerator = subdata[t,i,j]
		# 		for nni in range(nneighbors):
		# 			nnn = neighbors[nni]
		# 			numerator -= NB0[0,nnn,t]*psis[nni,i,j]
		# 		NBi[1,t] += numerator
		# NBi[1,t] /= float(nsx*nsy)

		#### this is the intensity
		NBi[0,t] = 0.
		for i in range(nsx):
			for j in range(nsy):
				numerator = subdata[t,i,j] - NB0[1,nmoli,t]
				for nni in range(nneighbors):
					nnn = neighbors[nni]
					if nnn != nmoli:
						numerator -= NB0[0,nnn,t]*psis[nni,i,j]
				for nni in range(nneighbors):
					nnn = neighbors[nni]
					if nnn == nmoli:
						numerator *= psis[nni,i,j]
				NBi[0,t] += numerator
		NBi[0,t] /= sumpsi2

	return NBi


if __name__ == '__main__':
	grid = gridclass(
		np.array([0.,0.]),
		np.array([1.,1.]),
		np.array([128,128])
	)

	nmol = 100
	sigma = 1.
	nsigma = 5
	offset = 0.5

	xys = np.random.rand(nmol,2)*(grid.nxy*grid.dxy+grid.origin)[None,:]
	weights = np.random.rand(nmol)*1.	
	z = density_atoms(grid,xys,weights,sigma,nsigma,offset)

	import matplotlib.pyplot as plt
	plt.imshow(z)
	plt.show()