#### Find Spots
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display,clear_output

from .. import spotfinder

fn_data = None
fn_align = None
fn_cal = None

def gui_spotfinder():
	out = widgets.Output()

	## initial guess to file
	default = None
	fns = os.listdir('./')
	for fn in fns:
		if fn.endswith('.tif'):
			default = fn
			break

	wl = widgets.Layout(width='80%',height='24pt')
	ws = {'description_width':'initial'}

	text_data_filename = widgets.Textarea(value=default,placeholder='Enter microscope data file name (.tif)',description="Data file name",layout=wl, style=ws)
	text_align_filename = widgets.Textarea(value='',placeholder='Enter alignment filen ame (.npy)',description="Alignment file name",layout=wl, style=ws)
	text_calibration_filename = widgets.Textarea(value='',placeholder='Enter calibration file name (.npy) [optional]',description="Calibration file name",layout=wl, style=ws)
	accordion_files = widgets.Accordion(children=[widgets.VBox([text_data_filename,text_align_filename,text_calibration_filename,]),], titles=('Files',))
	accordion_files.selected_index = 0

	dropdown_split = widgets.Dropdown(value='L/R',options=['L/R','T/B'],ensure_option=True,description='Split:', style=ws)
	int_acf_start_g = widgets.IntText(value=0,description='(Green) First Frame',style=ws)
	int_acf_end_g = widgets.IntText(value=0,description='(Green) Last Frame ', style=ws)
	int_acf_start_r = widgets.IntText(value=0,description='(Red) First Frame',style=ws)
	int_acf_end_r = widgets.IntText(value=0,description='(Red) Last Frame ',style=ws)
	accordion_acf = widgets.Accordion(children=[widgets.VBox([dropdown_split,int_acf_start_g,int_acf_end_g,int_acf_start_r,int_acf_end_r,]),], titles=('ACF',))
	# accordion_acf.selected_index = 0

	dropdown_localmax = widgets.Dropdown(value=1, options=[1,2,3,4,5,6,7,8,9,10,11],description='Local Max Radius (pixels)',style=ws)
	float_smooth = widgets.BoundedFloatText(value=.6,min=0,max=100,step=.1,description='Smoothing (pixels)',style=ws)
	dropdown_matched = widgets.Dropdown(value='True',options=['True','False'],ensure_option=True,description='Matched Spots?',style=ws)
	dropdown_which = widgets.Dropdown(value='Both',options=['Both','Only Green','Only Red'],ensure_option=True,description='Which Spots?',style=ws)
	float_acf_cutoff = widgets.BoundedFloatText(value=0.2,min=0,max=1,step=.001,description='Spotfinding ACF cutoff',style=ws)
	accordion_spots = widgets.Accordion(children=[widgets.VBox([dropdown_localmax,float_smooth,dropdown_which,dropdown_matched,float_acf_cutoff,]),], titles=('Spotfinding',))
	# accordion_spots.selected_index = 0

	button_prepare = widgets.Button(description="Prepare Data",layout=widgets.Layout(width='2in',height='0.25in'),style=ws)
	button_spotfind = widgets.Button(description="Find Spots",layout=widgets.Layout(width='2in',height='0.25in'),style=ws)

	vbox_spots = widgets.VBox([accordion_files,accordion_acf,accordion_spots,button_prepare,button_spotfind,])

	def show_prep_ui():
		with out:
			display(vbox_spots)

	def click_prepare(b):
		global fn_data,fn_align,fn_cal
		with out:
			out.clear_output()
			show_prep_ui()

			data_filename = re.sub(r'\s+', '', text_data_filename.value)
			align_filename = re.sub(r'\s+', '', text_align_filename.value)
			cal_filename = re.sub(r'\s+', '', text_calibration_filename.value)

			flags = spotfinder.default_flags()
			flags['split'] = dropdown_split.value
			flags['acf_cutoff'] = float_acf_cutoff.value
			flags['matched_spots'] = dropdown_matched.value == "True"
			flags['acf_start_g'] = int_acf_start_g.value
			flags['acf_end_g'] = int_acf_end_g.value
			flags['acf_start_r'] = int_acf_start_r.value
			flags['acf_end_r'] = int_acf_end_r.value
			flags['smooth'] = float_smooth.value
			flags['which'] = dropdown_which.value
			flags['localmax_region'] = dropdown_localmax.value

			# for key in flags.keys():
			# 	print(key,flags[key])
			# print(data_filename,align_filename,cal_filename)
			
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

			spotfinder.prepare_data(fn_data,fn_cal,flags)
			spotfinder.make_composite_aligned(fn_data,fn_align,flags)
			
			fig,ax = spotfinder.render_overlay(fn_data,fn_align,flags)
			fig.set_figheight(8.)
			fig.set_figwidth(8.)
			[plt.savefig(os.path.join(out_dir,'overlay_%s.%s'%(prefix,ext))) for ext in ['png','pdf']]
			plt.show()
		

	def click_spotfind(b):
		global fn_data,fn_align,fn_cal
		# clear_output()
		# show_prep_ui()
		with out:

			if fn_data is None or fn_align is None:
				print('Please prepare data first')
				return

			flags = spotfinder.default_flags()
			flags['split'] = dropdown_split.value
			flags['acf_cutoff'] = float_acf_cutoff.value
			flags['matched_spots'] = dropdown_matched.value == "True"
			flags['acf_start_g'] = int_acf_start_g.value
			flags['acf_end_g'] = int_acf_end_g.value
			flags['acf_start_r'] = int_acf_start_r.value
			flags['acf_end_r'] = int_acf_end_r.value
			flags['smooth'] = float_smooth.value
			flags['which'] = dropdown_which.value
			flags['localmax_region'] = dropdown_localmax.value

			out_dir = spotfinder.get_out_dir(fn_data)
			prefix = os.path.split(out_dir)[1][19:]

			g_spots_g,g_spots_r,r_spots_r,g_spots,r_spots = spotfinder.find_spots(fn_data,fn_align,flags)
			spotfinder.save_spots(prefix,fn_data,g_spots_g,g_spots_r,r_spots_r,g_spots,r_spots)
			
			fig,ax = spotfinder.render_found_maxes(fn_data,fn_align,g_spots_g,g_spots_r,flags)
			fig.set_figheight(8.)
			fig.set_figwidth(8.)
			[plt.savefig(os.path.join(out_dir,'spots_all_%s.%s'%(prefix,ext))) for ext in ['png','pdf']]
			plt.show()

			fig,ax = spotfinder.render_final_spots(fn_data,g_spots,r_spots,flags)
			fig.set_figheight(8.)
			fig.set_figwidth(8.)
			[plt.savefig(os.path.join(out_dir,'spots_final_%s.%s'%(prefix,ext))) for ext in ['png','pdf']]
			plt.show()

		
		

	button_prepare.on_click(click_prepare)	
	button_spotfind.on_click(click_spotfind)
	display(out)
	show_prep_ui()
	