import numpy as np
import numba as nb


@nb.njit
def med_huang_boundary(dd,w):
	nx,ny = dd.shape
	filtered = np.zeros_like(dd)
	histogram = np.zeros(256,dtype='int')
	w2 = int(w*w)
	r = int((w-1)//2)

	for i in range(nx):

		### zero histogram
		for hi in range(256):
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

@nb.njit
def med_perreault_boundary(dd,w):
	nx,ny = dd.shape
	filtered = np.zeros_like(dd)
	kernel_histogram = np.zeros(256,dtype='int')
	nn = 0
	column_histograms = np.zeros((ny,256),dtype='int')
	nnc = np.zeros(ny,dtype='int')
	
	w2 = int(w*w)
	r = int((w-1)//2)

	######### Initialize things
	i = 0
	j = 0
	
	##initialize column histograms
	for j in range(ny):
		for k in range(r+1):
			column_histograms[j,dd[k,j]] += 1
			nnc[j] += 1
	
	## initialize kernel histogram
	for l in range(r+1):
		kernel_histogram += column_histograms[l]
		nn += nnc[l]
	
	### first row doesn't get updates
	for j in range(ny):
		if j > 0:
			hjl = j - r - 1
			hjr = j + r
			if hjl >= 0:
				kernel_histogram -= column_histograms[hjl]
				nn -= nnc[hjl]
			if hjr < ny:
				kernel_histogram += column_histograms[hjr]
				nn += nnc[hjr]
			
		cut = int((nn//2)+1)
		count = 0
		
		for ci in range(kernel_histogram.size):
			count += kernel_histogram[ci]
			if count >= cut:
				filtered[i,j] = ci
				break

	######### Do Rows 
	for i in range(1,nx):
		for j in range(ny):
			## start the next row
			if j == 0:
				kernel_histogram *= 0
				nn = 0
				hit = i-r-1
				hib = i+r
				for l in range(r+1):
					if hit >= 0:
						column_histograms[l,dd[hit,l]] -= 1
						nnc[l] -= 1
					if hib < nx:
						column_histograms[l,dd[hib,l]] += 1
						nnc[l] += 1
					kernel_histogram += column_histograms[l]
					nn += nnc[l]
			
			## go through the row
			else:
				hit = i-r-1
				hib = i+r
				hjl = j-r-1
				hjr  = j+r

				#### update column histograms
				## top
				if hit >= 0:
					column_histograms[hjr,dd[hit,hjr]] -= 1
					nnc[hjr] -= 1

				## bottom
				if hib < nx:
					column_histograms[hjr,dd[hib,hjr]] += 1
					nnc[hjr] += 1

				#### update kernel histogram
				## left
				if hjl >= 0:
					kernel_histogram -= column_histograms[hjl]
					nn -= nnc[hjl]

				## right
				if hjr < ny:
					kernel_histogram += column_histograms[hjr]
					nn += nnc[hjr]
					
			
			## find median of kernel histogram
			cut = int((nn//2)+1)
			count = 0
			for ci in range(kernel_histogram.size):
				count += kernel_histogram[ci]
				if count >= cut:
					filtered[i,j] = ci
					break

	return filtered