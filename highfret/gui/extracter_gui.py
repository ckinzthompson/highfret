#### Extract
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display,clear_output

from .. import extracter,spotfinder

fn_data = None
fn_align = None
fn_cal = None

def gui_extracter():
	out = widgets.Output()

	# ## initial guess to file
	# default = None
	# fns = os.listdir('./')
	# for fn in fns:
	# 	if fn.endswith('.tif'):
	# 		default = fn
	# 		break
	default = None

	wl = widgets.Layout(width='80%',height='24pt')
	ws = {'description_width':'initial'}

	text_data_filename = widgets.Textarea(value=default,placeholder='Enter microscope data file name (.tif)',description="Data file name",layout=wl, style=ws)
	text_align_filename = widgets.Textarea(value='',placeholder='Enter alignment filen ame (.npy)',description="Alignment file name",layout=wl, style=ws)
	text_calibration_filename = widgets.Textarea(value='',placeholder='Enter calibration file name (.npy) [optional]',description="Calibration file name",layout=wl, style=ws)
	accordion_files = widgets.Accordion(children=[widgets.VBox([text_data_filename,text_align_filename,text_calibration_filename,]),], titles=('Files',))
	accordion_files.selected_index = 0

	dropdown_split = widgets.Dropdown(value='L/R',options=['L/R','T/B'],ensure_option=True,description='Split:', style=ws)
	dropdown_dl = widgets.Dropdown(value=5, options=[1,2,3,4,5,6,7,8,9,10,11],description='Extraction Radius (pixels)',style=ws)
	float_pixel_real = widgets.BoundedFloatText(value=6500.,min=1,max=1000000,step=.1,description='Pixel Length (nm)',style=ws)
	float_mag = widgets.BoundedFloatText(value=60.,min=1,max=1000000,step=.1,description='Magnification (x)',style=ws)
	dropdown_bin = widgets.Dropdown(value=2., options=[1.,2.,4.,8.],description='Binning',style=ws)
	float_lambda_nm = widgets.BoundedFloatText(value=580.,min=1,max=1000000,step=.1,description='Wavelength (nm)',style=ws)
	float_NA= widgets.BoundedFloatText(value=1.2,min=0.,max=1000000,step=.2,description='Numerical Aperture',style=ws)
	float_motion = widgets.BoundedFloatText(value=100. ,min=0.,max=1000000,step=.2,description='Motion RMSD (nm)',style=ws)

	float_sigma = widgets.BoundedFloatText(value=.85,min=0,max=1000000,step=.01,description='PSF width (pixels)',style=ws)

	int_nsigma = widgets.BoundedIntText(value=61,min=1,max=1000000,description='Number of Sigmas',layout=wl,style=ws)
	range_minmaxsigma = widgets.FloatRangeSlider(value=[.2, 2.],min=0,max=10,step=.01,description='Sigma Limits:',orientation='horizontal',readout_format='.2f',layout=wl,style=ws)
	dropdown_keeptbbins = widgets.Dropdown(value='False',options=['True','False'],ensure_option=True,description='Keep First/Last Bins:', layout=wl, style=ws)
	dropdown_optmethod= widgets.Dropdown(value='ACF',options=['All','ACF','Max','Mean'],ensure_option=True,description='Data Treatment:', layout=wl, style=ws)
	int_opt_start = widgets.IntText(value=0,description='First Frame', layout=wl,style=ws)
	int_opt_end = widgets.IntText(value=0,description='Last Frame', layout=wl,style=ws)
	button_optimize = widgets.Button(description='Optimize Sigma',layout=widgets.Layout(width='2in',height='0.25in'),style=ws)

	tab_microscope = widgets.Tab(description='')
	tab_microscope.children = [
		widgets.VBox([dropdown_split,dropdown_dl,float_sigma]),
		widgets.VBox([dropdown_split,dropdown_dl,float_pixel_real,float_mag,dropdown_bin,float_lambda_nm,float_NA,float_motion,]),
		widgets.VBox([dropdown_split,dropdown_dl,int_nsigma,range_minmaxsigma,dropdown_keeptbbins,dropdown_optmethod,int_opt_start,int_opt_end,button_optimize]),
	]
	tab_microscope.titles = ['Simple','Advanced','Optimize']

	button_extract = widgets.Button(description="Extract",layout=widgets.Layout(width='2in',height='0.25in'),style=ws)
	vbox_extract = widgets.VBox([accordion_files, tab_microscope, button_extract,])

	def show_prep_ui():
		with out:
			out.clear_output()
			display(vbox_extract)

	def click_optimize(b):
		global fn_data,fn_align,fn_cal
		with out:
			show_prep_ui()

			data_filename = re.sub(r'\s+', '', text_data_filename.value)
			align_filename = re.sub(r'\s+', '', text_align_filename.value)
			cal_filename = re.sub(r'\s+', '', text_calibration_filename.value)

			
			split = dropdown_split.value
			dl = int(dropdown_dl.value)

			if cal_filename == '':
				check_these = [data_filename,align_filename]
				print('Ignoring Calibration')
			else:
				check_these = [data_filename,align_filename,cal_filename]

			for fn in check_these:
				if os.path.exists(fn):
					print("Found: %s"%(fn))
				else:
					print('Failure: File does not exist !!!! %s'%(fn))
					fn_data = None
					fn_align = None
					fn_cal = None
					return
			
			fn_data = data_filename
			fn_align = align_filename
			fn_cal = cal_filename if cal_filename != '' else None

			out_dir = spotfinder.get_out_dir(fn_data)
			prefix = os.path.split(out_dir)[1][19:]

			nsigma = int_nsigma.value
			sigma_low,sigma_high = range_minmaxsigma.value
			flag_keeptbbins = dropdown_keeptbbins.value == "True"
			method = dropdown_optmethod.value
			first = int_opt_start.value
			last = int_opt_end.value
			fig,ax = extracter.optimize_sigma(fn_data,fn_align,fn_cal,split,dl,nsigma,sigma_low,sigma_high,flag_keeptbbins,method,first,last)
			[plt.savefig(os.path.join(out_dir,'sigma_optimization_%s.%s'%(prefix,ext))) for ext in ['png','pdf']]
			plt.show()

	def click_extract(b):
		global fn_data,fn_align,fn_cal
		with out:
			show_prep_ui()

			data_filename = re.sub(r'\s+', '', text_data_filename.value)
			align_filename = re.sub(r'\s+', '', text_align_filename.value)
			cal_filename = re.sub(r'\s+', '', text_calibration_filename.value)

			
			split = dropdown_split.value
			dl = int(dropdown_dl.value)
			if tab_microscope.selected_index == 0:
				sigma = float(float_sigma.value)
			elif tab_microscope.selected_index == 1:
				pixel_real = float(float_pixel_real.value)
				mag = float(float_mag.value)
				bin = float(dropdown_bin.value)
				lambda_nm = float(float_lambda_nm.value)
				NA = float(float_NA.value)
				motion = float(float_motion.value)
				sigma = float(extracter.calculate_sigma(pixel_real,mag,bin,lambda_nm,NA,motion))
			
			if cal_filename == '':
				check_these = [data_filename,align_filename]
				print('Ignoring Calibration')
			else:
				check_these = [data_filename,align_filename,cal_filename]

			for fn in check_these:
				if os.path.exists(fn):
					print("Found: %s"%(fn))
				else:
					print('Failure: File does not exist !!!! %s'%(fn))
					fn_data = None
					fn_align = None
					fn_cal = None
					return
			
			fn_data = data_filename
			fn_align = align_filename
			fn_cal = cal_filename if cal_filename != '' else None

			out_dir = spotfinder.get_out_dir(fn_data)
			prefix = os.path.split(out_dir)[1][19:]
			
			dg,dr = extracter.prepare_data(fn_data,fn_align,fn_cal,split)
			spots_g,spots_r = extracter.load_spots(fn_data)
			intensities = extracter.get_intensities(dg,dr,spots_g,spots_r,dl,sigma)
			
			extracter.write_hdf5(fn_data,intensities)

			fig,ax = plt.subplots(1)
			ax.plot(np.nanmean(intensities,axis=0)[:,0],color='tab:green',lw=1)
			ax.plot(np.nanmean(intensities,axis=0)[:,1],color='tab:red',lw=1)
			ax.set_xlabel('Time (frame)')
			ax.set_ylabel('Average Intensity')
			fig.set_figheight(6.)
			fig.set_figwidth(6.)
			[plt.savefig(os.path.join(out_dir,'intensity_avg_%s.%s'%(prefix,ext))) for ext in ['png','pdf']]
			plt.show()

	button_optimize.on_click(click_optimize)	
	button_extract.on_click(click_extract)
	show_prep_ui()
	display(out)