import time
import numpy as np
import matplotlib.pyplot as plt

from .containers import general_analysis_class
from .support.fast_median import median_scmos, med_huang_floatimg
from .support.prepare import acf

def prepare_img(analysis: general_analysis_class, start: int, end: int = 0, median: int=0):
	if end == 0:
		end = analysis.data.shape[1]
	
	analysis.log += f'Image Processing:\n\tAutocorrelation Function Image (tau = 1 frame)\n'
	analysis.log += f'\tStart Frame: {start}\n'
	analysis.log += f'\tEnd Frame: {end}\n'

	t0 = time.time()
	analysis.img = np.zeros((analysis.data.shape[0], analysis.data.shape[-2], analysis.data.shape[-1]))
	for i in range(analysis.img.shape[0]):
		analysis._img[i] = acf(analysis.data[i,start:end])
		if median > 0:
			analysis._img[i] -= med_huang_floatimg(analysis._img[i],median)
	t1 = time.time()
	analysis.log += f'\tTime Elapsed: {t1-t0:.3f}s\n'

def filter_movie(analysis: general_analysis_class,median_filter:int=3):
	analysis.data.shape ## force it to load
	for i in range(analysis._data.shape[0]):
		analysis._data[i] = median_scmos(analysis._data[i],median_filter)
	analysis.log += f'Median Filtered movie (kernel={median_filter})\n'

def plot_avg_intensity(analysis: general_analysis_class,):
	color_cycle = ['tab:green','tab:red']
	fig,ax = plt.subplots(1)
	for i in range(analysis.data.shape[0]):
		if i < len(color_cycle):
			ax.plot(analysis.data[i].mean((1,2)),label=f'Color {i}',color=color_cycle[i])
		else:
			ax.plot(analysis.data[i].mean((1,2)),label=f'Color {i}')
	ax.legend()
	return fig

