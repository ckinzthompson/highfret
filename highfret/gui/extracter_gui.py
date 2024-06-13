#### Extract
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display,clear_output

from .. import extracter,spotfinder,prepare

fn_data = None
fn_align = None
fn_cal = None

def gui_extracter():
	out = widgets.Output()

	default = extracter.default_flags()

	wl = widgets.Layout(width='80%',height='24pt')
	ws = {'description_width':'initial'}

	text_data_filename = widgets.Textarea(value=default['fn_data'],placeholder='Enter microscope data file name (.tif)',description="Data file name",layout=wl, style=ws)
	text_align_filename = widgets.Textarea(value=default['fn_align'],placeholder='Enter alignment file name (.npy)',description="Alignment file name",layout=wl, style=ws)
	text_calibration_filename = widgets.Textarea(value=default['fn_cal'],placeholder='Enter calibration file name (.npy) [optional]',description="Calibration file name",layout=wl, style=ws)
	accordion_files = widgets.Accordion(children=[widgets.VBox([text_data_filename,text_align_filename,text_calibration_filename,]),], titles=('Files',))
	accordion_files.selected_index = 0

	dropdown_method = widgets.Dropdown(value=default['method'], options=['MLE PSF','Mask (4px)'], ensure_option=True,description='Method:',style=ws)
	dropdown_split = widgets.Dropdown(value=default['split'],options=['L/R','T/B'],ensure_option=True,description='Split:', style=ws)
	dropdown_dl = widgets.Dropdown(value=default['dl'], options=[1,2,3,4,5,6,7,8,9,10,11],description='Extraction Radius (pixels)',style=ws)
	float_pixel_real = widgets.BoundedFloatText(value=default['pixel_real'],min=1,max=1000000,step=.1,description='Pixel Length (nm)',style=ws)
	float_mag = widgets.BoundedFloatText(value=default['mag'],min=1,max=1000000,step=.1,description='Magnification (x)',style=ws)
	dropdown_bin = widgets.Dropdown(value=default['bin'], options=[1.,2.,4.,8.],description='Binning',style=ws)
	float_lambda_nm = widgets.BoundedFloatText(value=default['lambda_nm'],min=1,max=1000000,step=.1,description='Wavelength (nm)',style=ws)
	float_NA= widgets.BoundedFloatText(value=default['NA'],min=0.,max=1000000,step=.2,description='Numerical Aperture',style=ws)
	float_motion = widgets.BoundedFloatText(value=default['motion'],min=0.,max=1000000,step=.2,description='Motion RMSD (nm)',style=ws)

	float_sigma = widgets.BoundedFloatText(value=default['sigma'],min=0,max=1000000,step=.01,description='PSF width (pixels)',style=ws)

	int_nsigma = widgets.BoundedIntText(value=default['nsigma'],min=1,max=1000000,description='Number of Sigmas',layout=wl,style=ws)
	range_minmaxsigma = widgets.FloatRangeSlider(value=[default['sigma_low'], default['sigma_high']],min=0,max=10,step=.01,description='Sigma Limits:',orientation='horizontal',readout_format='.2f',layout=wl,style=ws)
	sbool = 'True' if default['flag_keeptbbins'] else 'False'
	dropdown_keeptbbins = widgets.Dropdown(value=sbool,options=['True','False'],ensure_option=True,description='Keep First/Last Bins:', layout=wl, style=ws)
	dropdown_optmethod= widgets.Dropdown(value=default['optmethod'],options=['All','ACF','Max','Mean'],ensure_option=True,description='Data Treatment:', layout=wl, style=ws)
	int_opt_start = widgets.IntText(value=default['first'],description='First Frame', layout=wl,style=ws)
	int_opt_end = widgets.IntText(value=default['last'],description='Last Frame', layout=wl,style=ws)
	button_optimize = widgets.Button(description='Optimize Sigma',layout=widgets.Layout(width='2in',height='0.25in'),style=ws)

	tab_microscope = widgets.Tab(description='')
	tab_microscope.children = [
		widgets.VBox([dropdown_split,dropdown_method,dropdown_dl,float_sigma]),
		widgets.VBox([dropdown_split,dropdown_method,dropdown_dl,float_pixel_real,float_mag,dropdown_bin,float_lambda_nm,float_NA,float_motion,]),
		widgets.VBox([dropdown_split,dropdown_dl,int_nsigma,range_minmaxsigma,dropdown_keeptbbins,dropdown_optmethod,int_opt_start,int_opt_end,button_optimize]),
	]
	tab_microscope.titles = ['Simple','Advanced','Optimize']

	button_extract = widgets.Button(description="Extract",layout=widgets.Layout(width='2in',height='0.25in'),style=ws)
	vbox_extract = widgets.VBox([accordion_files, tab_microscope, button_extract,])

	def show_prep_ui():
		with out:
			out.clear_output()
			display(vbox_extract)

	def generate_job():
		global fn_data,fn_align,fn_cal

		data_filename = re.sub(r'\s+', '', text_data_filename.value)
		align_filename = re.sub(r'\s+', '', text_align_filename.value)
		cal_filename = re.sub(r'\s+', '', text_calibration_filename.value)

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

		job = extracter.default_flags()

		job['split'] = dropdown_split.value
		job['dl'] = int(dropdown_dl.value)
		job['method'] = dropdown_method.value
		job['nsigma'] = int_nsigma.value
		job['sigma_low'] = range_minmaxsigma.value[0]
		job['sigma_high'] = range_minmaxsigma.value[1]
		job['flag_keeptbbins'] = dropdown_keeptbbins.value == "True"
		job['optmethod'] = dropdown_optmethod.value
		job['first'] = int_opt_start.value
		job['last'] = int_opt_end.value

		if tab_microscope.selected_index == 0:
			job['sigma'] = float(float_sigma.value)
		elif tab_microscope.selected_index == 1:
			pixel_real = float(float_pixel_real.value)
			mag = float(float_mag.value)
			bin = float(dropdown_bin.value)
			lambda_nm = float(float_lambda_nm.value)
			NA = float(float_NA.value)
			motion = float(float_motion.value)
			job['sigma'] = float(extracter.calculate_sigma(pixel_real,mag,bin,lambda_nm,NA,motion))

		job['fn_data'] = fn_data
		job['fn_align'] = fn_align
		job['fn_cal'] = fn_cal

		return job

	def click_optimize(b):
		with out:
			show_prep_ui()
			job = generate_job()
			fig = extracter.run_job_optimize(job)
			plt.show()

	def click_extract(b):
		with out:
			show_prep_ui()
			job = generate_job()
			fig = extracter.run_job_extract(job)
			plt.show()

	button_optimize.on_click(click_optimize)	
	button_extract.on_click(click_extract)
	show_prep_ui()
	display(out)