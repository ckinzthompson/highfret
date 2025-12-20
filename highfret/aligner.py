import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .support.aligner import initialize_theta,optimize_data,render_images
from .support import modelselect_alignment as alignment
from .containers import general_analysis_class	

def clear(analysis: general_analysis_class):
	ncolors = analysis.img.shape[0]
	_base_transform = alignment.coefficients_blank(1)
	analysis._transforms = np.array([[_base_transform for j in range(ncolors)] for i in range(ncolors)])

def initialize(analysis: general_analysis_class):
	analysis.log += 'Alignment: Initializing with Fourier guess\n'
	ncolors = analysis.img.shape[0]
	order = 1
	clear(analysis)
	for i in range(ncolors):
		for j in range(ncolors):
			if i == j:
				analysis.transforms[i,j] = alignment.coefficients_blank(order)
			elif i > j:
				analysis.transforms[i,j] = alignment.invert_transform(analysis.transforms[j,i],analysis.img[0].shape,nl=100)
			elif i < j:
				theta,log = initialize_theta(analysis.img[i],analysis.img[j])
				analysis.transforms[i,j] = theta
				analysis.log += log

def optimize(analysis: general_analysis_class, order: int = 1, downscale: float = 1., maxiter: int = 5, miniter: int= 1):
	analysis.log += 'Alignment: Optimizing\n'
	analysis.log += f'\tPolynomial Order: {order}\n'
	analysis.log += f'\tDownscale Factor: {downscale}\n'
	analysis.log += f'\tMinimum Iterations: {miniter}\n'
	analysis.log += f'\tMaximum Iterations: {maxiter}\n'
	ncolors = analysis.img.shape[0]
	_base_transform = alignment.coefficients_blank(order)
	old_transforms = analysis._transforms.copy()
	analysis._transforms = np.array([[_base_transform for j in range(ncolors)] for i in range(ncolors)])
	for i in range(ncolors):
		for j in range(ncolors):
			if i == j:
				analysis.transforms[i,j] = alignment.coefficients_blank(order)
			if i > j:
				analysis.transforms[i,j] = alignment.invert_transform(analysis.transforms[j,i],analysis.img[0].shape,nl=100)
			if i < j:
				theta,log = optimize_data(analysis.img[i], analysis.img[j], old_transforms[i,j], order, downscale, maxiter, miniter)
				analysis.transforms[i,j] = theta
				analysis.log += log

def render(analysis: general_analysis_class):
	ncolors = analysis.img.shape[0]
	figaxs = []
	for i in range(ncolors):
		for j in range(ncolors):
			if i!=j:
				## img j destination, img i original, reverse transform (theta_ji)
				fig,ax = render_images(analysis.img[j], analysis.img[i], analysis.transforms[j,i]) ## nb, transform should be the reverse need ji to make ij image!!!
				figaxs.append([fig,ax])
	return figaxs
