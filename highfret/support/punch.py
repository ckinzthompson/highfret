import numba as nb
import numpy as np

@nb.njit(cache=True)
def get_punches(img,spots,l=11,fill_value=0.):
	nx,ny = img.shape
	ns,_ = spots.shape
	nl = 2*l+1

	out = np.zeros((ns,nl,nl)) + fill_value
	for i in range(ns):
		mx = int(spots[i,0])
		my = int(spots[i,1])

		xmin = int(max(0,mx-l))
		xmax = int(min(nx,mx+l+1))

		ymin = int(max(0,my-l))
		ymax = int(min(ny,my+l+1))

		if xmin >= xmax or ymin >= ymax:
			continue

		out[i,l-(mx-xmin):l+(xmax-mx),l-(my-ymin):l+(ymax-my)] = img[xmin:xmax,ymin:ymax]
	return out

@nb.njit(cache=True)
def tile_punches(punches):
	ns,l1,l2 = punches.shape
	ntiles = int(np.sqrt(ns))
	if ntiles**2 != ns:
		ntiles += 1

	img = np.zeros(((1+l1)*ntiles,(1+l2)*ntiles)) + punches.mean()
	for i in range(ntiles):
		for j in range(ntiles):
			if i*ntiles+j < ns:
				img[i*(l1+1)+1:(i+1)*(l1+1),j*(1+l2)+1:(j+1)*(1+l2)] = punches[i*ntiles+j]
			img[i*(l1+1),:] = np.nan
			img[:,j*(1+l2)] = np.nan
	return img

def argsort_punches(punches):
	ns,nx,ny = punches.shape
	xsort = punches[:,nx//2,ny//2].argsort()[::-1]
	return xsort

def sort_punches(punches):
	xsort = argsort_punches(punches)
	return punches[xsort]