import numpy as np
import numba as nb

#### See "Median Filtering in Constant Time" by Simon Perreault and Patrick Hebert for more details on Huang and Perreault algorithm
#### Also checkout: http://ckinzthompson.github.io/posts/2023-10-29-median.html

def median_scmos(d,box_side=2*10+1,bitdepth=12,filter=True):
	'''
	d is a (t,x,y) uint16 numpy ndarray. 
	box_side defines the median region. i.e., box_side=3 means a 9 pixel region.
	bitdepth is how far down to perform the median calculation; it should be 12 for scmos cameras (see below)
	filter: if true, then the returned data is `d-median`, else it's just median

	note: these are 12-bit cameras stored in 16bit tif files, and usually they just bit shift << 4, so 
	here, the bit shifting happens inside the function
	'''
	if d.dtype != 'uint16':
		raise Exception('Expected a uint16 image')
	if d.ndim == 2:
		return _median_huang_scmos(d[None,:,:], box_side, bitdepth, filter)
	elif d.ndim == 3:
		return _median_huang_scmos(d, box_side, bitdepth, filter)
	else:
		raise Exception(f'The data shape is weird {d.shape}')
	
@nb.njit(nogil=True,parallel=True,fastmath=True,cache=True)
def _median_huang_scmos(d,w,bitdepth=12,filter=True):
	### this is very similar to med_huang_boundary, but works on a movie and has hardcoded bit shifting
	if d.ndim != 3:
		raise Exception('Need a uint16 (t,x,y) movie')
	nt,nx,ny = d.shape
	filtered = np.zeros((nt,nx,ny),dtype='uint16')
	
	w2 = int(w*w)
	r = int((w-1)//2)
	dbd = 16-bitdepth

	for t in nb.prange(nt):
	# for t in range(nt):
		histogram = np.zeros(2**bitdepth,dtype='int')
		for i in range(nx):

			### zero histogram
			for hi in range(histogram.size):
				histogram[hi] = 0
			
			### initialize histogram
			j = 0
			nn = 0
			for k in range(-r,r+1):
				for l in range(-r,r+1):
					hi = i+k
					hj = j+l
					if hi >=0 and hj>=0 and hi<nx and hj<ny:
						histogram[d[t,hi,hj]>>dbd] += 1 ## shift down to 16-bitdepth
						nn += 1

			## find median of histogram
			cut = int((nn//2)+1)
			count = 0
			for ci in range(histogram.size):
				count += histogram[ci]
				if count >= cut:
					if filter:
						temp = ci<<dbd ## shift up back to 16
						if d[t,i,j] > temp:
							filtered[t,i,j] = d[t,i,j] - temp
						else:
							filtered[t,i,j] = 0
					else:
						filtered[t,i,j] = ci<<dbd ## shift up back to 16
					break

			### run row
			for j in range(1,ny):
				hjl = j-r -1
				hjr = j+r
					
				for k in range(-r,r+1):
					hi = i+k
					## add RHS histogram
					if hi >=0 and hi<nx and hjr<ny:
						histogram[d[t,hi,hjr]>>dbd] += 1 ## shift down to 16-bitdepth
						nn += 1
					## remove LHS histogram
					if hi >=0 and hjl>=0 and hi<nx:
						histogram[d[t,hi,hjl]>>dbd] -= 1 ## shift down to 16-bitdepth
						nn -= 1

				## find median of histogram
				cut = int((nn//2)+1)
				count = 0
				for ci in range(histogram.size):
					count += histogram[ci]
					if count >= cut:
						if filter:
							temp = ci<<dbd ## shift up back to 16
							if d[t,i,j] > temp:
								filtered[t,i,j] = d[t,i,j] - temp
							else:
								filtered[t,i,j] = 0
						else:
							filtered[t,i,j] = ci<<dbd ## shift up back to 16
						break
	return filtered


######## For posterity.

@nb.njit(cache=True)
def med_huang_floatimg(dd,w):
	bitdepth=12
	nx,ny = dd.shape
	q = np.zeros((nx,ny),'uint16')
	out = np.zeros((nx,ny),'double')
	
	dmax = np.max(dd)
	dmin = np.min(dd)
	bmax = 2**bitdepth

	for i in range(nx):
		for j in range(ny):
			q[i,j] = int((dd[i,j] - dmin)/(dmax-dmin) * bmax)
	
	med = med_huang_boundary(q,w,bitdepth)
	
	for i in range(nx):
		for j in range(ny):
			out[i,j] = float(med[i,j])/bmax * (dmax-dmin) + dmin
	return out

@nb.njit(cache=True)
def med_huang_boundary(dd,w,bitdepth=8):
	nx,ny = dd.shape
	filtered = np.zeros_like(dd)
	histogram = np.zeros(2**bitdepth,dtype='int')
	w2 = int(w*w)
	r = int((w-1)//2)

	for i in range(nx):

		### zero histogram
		for hi in range(histogram.size):
			histogram[hi] = 0
		
		### initialize histogram
		j = 0
		nn = 0
		for k in range(-r,r+1):
			for l in range(-r,r+1):
				hi = i+k
				hj = j+l
				if hi >=0 and hj>=0 and hi<nx and hj<ny:
					histogram[dd[hi,hj]] += 1
					nn += 1

		## find median of histogram
		cut = int((nn//2)+1)
		count = 0
		for ci in range(histogram.size):
			count += histogram[ci]
			if count >= cut:
				filtered[i,j] = ci
				break

		### run row
		for j in range(1,ny):
			hjl = j-r -1
			hjr = j+r
				
			for k in range(-r,r+1):
				hi = i+k
				## add RHS histogram
				if hi >=0 and hi<nx and hjr<ny:
					histogram[dd[hi,hjr]] += 1
					nn += 1
				## remove LHS histogram
				if hi >=0 and hjl>=0 and hi<nx:
					histogram[dd[hi,hjl]] -= 1
					nn -= 1

			## find median of histogram
			cut = int((nn//2)+1)
			count = 0
			for ci in range(histogram.size):
				count += histogram[ci]
				if count >= cut:
					filtered[i,j] = ci
					break
	return filtered


# @nb.njit
# def med_perreault_boundary(dd,w,bitdepth=8):
# 	nx,ny = dd.shape
# 	filtered = np.zeros_like(dd)
# 	kernel_histogram = np.zeros(2**bitdepth,dtype='int')
# 	nn = 0
# 	column_histograms = np.zeros((ny,2**bitdepth),dtype='int')
# 	nnc = np.zeros(ny,dtype='int')
	
# 	w2 = int(w*w)
# 	r = int((w-1)//2)


# 	######### Initialize things
# 	i = 0
# 	j = 0
	
# 	##initialize column histograms
# 	for j in range(ny):
# 		for k in range(r+1):
# 			column_histograms[j,dd[k,j]] += 1
# 			nnc[j] += 1
	
# 	## initialize kernel histogram
# 	for l in range(r+1):
# 		kernel_histogram += column_histograms[l]
# 		nn += nnc[l]
	
# 	### first row doesn't get updates
# 	for j in range(ny):
# 		if j > 0:
# 			hjl = j - r - 1
# 			hjr = j + r
# 			if hjl >= 0:
# 				kernel_histogram -= column_histograms[hjl]
# 				nn -= nnc[hjl]
# 			if hjr < ny:
# 				kernel_histogram += column_histograms[hjr]
# 				nn += nnc[hjr]
			
# 		cut = int((nn//2)+1)
# 		count = 0
		
# 		for ci in range(kernel_histogram.size):
# 			count += kernel_histogram[ci]
# 			if count >= cut:
# 				filtered[i,j] = ci
# 				break

# 	######### Do Rows 
# 	for i in range(1,nx):
# 		for j in range(ny):
# 			## start the next row
# 			if j == 0:
# 				kernel_histogram *= 0
# 				nn = 0
# 				hit = i-r-1
# 				hib = i+r
# 				for l in range(r+1):
# 					if hit >= 0:
# 						column_histograms[l,dd[hit,l]] -= 1
# 						nnc[l] -= 1
# 					if hib < nx:
# 						column_histograms[l,dd[hib,l]] += 1
# 						nnc[l] += 1
# 					kernel_histogram += column_histograms[l]
# 					nn += nnc[l]
			
# 			## go through the row
# 			else:
# 				hit = i-r-1
# 				hib = i+r
# 				hjl = j-r-1
# 				hjr  = j+r

# 				#### update column histograms
# 				## top
# 				if hit >= 0 and hjr < ny:
# 					column_histograms[hjr,dd[hit,hjr]] -= 1
# 					nnc[hjr] -= 1

# 				## bottom
# 				if hib < nx and hjr < ny:
# 					column_histograms[hjr,dd[hib,hjr]] += 1
# 					nnc[hjr] += 1

# 				#### update kernel histogram
# 				## left
# 				if hjl >= 0:
# 					kernel_histogram -= column_histograms[hjl]
# 					nn -= nnc[hjl]

# 				## right
# 				if hjr < ny:
# 					kernel_histogram += column_histograms[hjr]
# 					nn += nnc[hjr]
					
# 			## find median of kernel histogram
# 			cut = int((nn//2)+1)
# 			count = 0
# 			for ci in range(kernel_histogram.size):
# 				count += kernel_histogram[ci]
# 				if count >= cut:
# 					filtered[i,j] = ci
# 					break

# 	return filtered