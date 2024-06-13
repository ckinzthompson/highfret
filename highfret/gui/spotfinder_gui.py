#### Find Spots
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display,clear_output

from .. import spotfinder,prepare

fn_data = None
fn_align = None
fn_cal = None

def gui_spotfinder():
	out = widgets.Output()

	default = spotfinder.default_flags()

	wl = widgets.Layout(width='80%',height='24pt')
	ws = {'description_width':'initial'}

	text_data_filename = widgets.Textarea(value=default['fn_data'],placeholder='Enter microscope data file name (.tif)',description="Data file name",layout=wl, style=ws)
	text_align_filename = widgets.Textarea(value=default['fn_align'],placeholder='Enter alignment file name (.npy)',description="Alignment file name",layout=wl, style=ws)
	text_calibration_filename = widgets.Textarea(value=default['fn_cal'],placeholder='Enter calibration file name (.npy) [optional]',description="Calibration file name",layout=wl, style=ws)
	accordion_files = widgets.Accordion(children=[widgets.VBox([text_data_filename,text_align_filename,text_calibration_filename,]),], titles=('Files',))

	dropdown_split = widgets.Dropdown(value=default['split'],options=['L/R','T/B'],ensure_option=True,description='Split:', style=ws)
	int_acf_start_g = widgets.IntText(value=default['acf_start_g'],description='(Green) First Frame',style=ws)
	int_acf_end_g = widgets.IntText(value=default['acf_end_g'],description='(Green) Last Frame ', style=ws)
	int_acf_start_r = widgets.IntText(value=default['acf_start_r'],description='(Red) First Frame',style=ws)
	int_acf_end_r = widgets.IntText(value=default['acf_end_r'],description='(Red) Last Frame ',style=ws)
	accordion_acf = widgets.Accordion(children=[widgets.VBox([dropdown_split,int_acf_start_g,int_acf_end_g,int_acf_start_r,int_acf_end_r,]),], titles=('ACF',))

	dropdown_localmax = widgets.Dropdown(value=default['localmax_region'], options=[1,2,3,4,5,6,7,8,9,10,11],description='Local Max Radius (pixels)',style=ws)
	float_smooth = widgets.BoundedFloatText(value=default['smooth'],min=0,max=100,step=.1,description='Smoothing (pixels)',style=ws)
	sbool = 'True' if default['matched_spots'] else 'False'
	dropdown_matched = widgets.Dropdown(value=sbool,options=['True','False'],ensure_option=True,description='Matched Spots?',style=ws)
	dropdown_which = widgets.Dropdown(value=default['which'],options=['Both','Only Green','Only Red'],ensure_option=True,description='Which Spots?',style=ws)
	float_acf_cutoff = widgets.BoundedFloatText(value=default['acf_cutoff'],min=0,max=1,step=.001,description='Spotfinding ACF cutoff',style=ws)
	sbool = 'True' if default['refine'] else 'False'
	checkbox_refine = widgets.Checkbox(value=sbool,description='Refine spots',layout=wl,style=ws)
	accordion_spots = widgets.Accordion(children=[widgets.VBox([dropdown_localmax,float_smooth,dropdown_which,dropdown_matched,float_acf_cutoff,checkbox_refine,]),], titles=('Spotfinding',))

	button_prepare = widgets.Button(description="Prepare Data",layout=widgets.Layout(width='2in',height='0.25in'),style=ws)
	button_spotfind = widgets.Button(description="Find Spots",layout=widgets.Layout(width='2in',height='0.25in'),style=ws)

	vbox_spots = widgets.VBox([accordion_files,accordion_acf,accordion_spots,button_prepare,button_spotfind,])

	def show_prep_ui():
		with out:
			display(vbox_spots)

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

		job = spotfinder.default_flags()
		job['split'] = dropdown_split.value
		job['acf_cutoff'] = float_acf_cutoff.value
		job['matched_spots'] = dropdown_matched.value == "True"
		job['acf_start_g'] = int_acf_start_g.value
		job['acf_end_g'] = int_acf_end_g.value
		job['acf_start_r'] = int_acf_start_r.value
		job['acf_end_r'] = int_acf_end_r.value
		job['smooth'] = float_smooth.value
		job['which'] = dropdown_which.value
		job['localmax_region'] = dropdown_localmax.value
		job['refine'] = checkbox_refine.value

		job['fn_data'] = fn_data
		job['fn_align'] = fn_align
		job['fn_cal'] = fn_cal

		return job

	def click_prepare(b):
		with out:
			out.clear_output()
			show_prep_ui()

			job = generate_job()
			fig = spotfinder.run_job_prepare(job)
			plt.show()

	def click_spotfind(b):
		global fn_data,fn_align,fn_cal
		with out:
			out.clear_output()
			show_prep_ui()

			if fn_data is None or fn_align is None:
				print('Please prepare data first')
				return
			
			job = generate_job()
			fig,fig2 = spotfinder.run_job_spotfind(job)
			plt.show()

	button_prepare.on_click(click_prepare)	
	button_spotfind.on_click(click_spotfind)
	display(out)
	show_prep_ui()


