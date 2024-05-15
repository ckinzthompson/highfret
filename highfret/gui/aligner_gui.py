#### Prepare
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display,clear_output

from .. import aligner

fn_data = None
results_index = None
results_names = []
results_thetas = []
out_dir = None

fig = None
ax = None


def gui_aligner():
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
	bl = layout=widgets.Layout(width='2in',height='0.25in')
	
	## Load widgets
	text_filename = widgets.Textarea(value=default,placeholder='Enter file to align',description="File name", layout=wl, style=ws)
	button_prepare = widgets.Button(description="Prepare Data", layout=bl)
	int_start = widgets.IntText(value=0,description='First Frame', style=ws)
	int_end = widgets.IntText(value=0,description='Last Frame ', style=ws)
	dropdown_method = widgets.Dropdown(value='mean',options=['mean','acf','First Frame'],ensure_option=True,description='Method:', style=ws)
	dropdown_split = widgets.Dropdown(value='L/R',options=['L/R','T/B'],ensure_option=True,description='Split:', style=ws)
	vbox_prepare = widgets.VBox([text_filename,int_start,int_end,dropdown_method,dropdown_split,button_prepare,])

	## Results widgets
	text_thetafilename = widgets.Textarea(value=None,placeholder='Enter Theta File Name',description="File Name", layout=wl, style=ws)
	button_loadtheta = widgets.Button(description="Load Theta", layout=bl)
	button_blanktheta = widgets.Button(description="Blank Theta", layout=bl)
	button_fouriertheta = widgets.Button(description="Estimate Theta (Fourier)", layout=bl)
	accordion_initializetheta = widgets.Accordion(children=[widgets.VBox([text_thetafilename,button_loadtheta,button_blanktheta,button_fouriertheta,]),], titles=('Initialize Theta',))

	# dropdown_results = widgets.Dropdown(value=results_names[results_index], options=results_names[::-1])
	dropdown_results = widgets.Dropdown(value=None, options=[],description='Results:',layout=wl,style=ws)

	dropdown_order = widgets.Dropdown(value=1, options=[1,2,3,4,5],description='Poly. Order', style=ws)
	dropdown_downscale = widgets.Dropdown(value=1, options=[1,2,4,8],description='Downscale', style=ws)
	range_minmaxiter = widgets.IntRangeSlider(value=[5, 10],min=0,max=10,step=1,description='Iteration Limits:',orientation='horizontal',readout_format='d',layout=wl,style=ws)
	button_optimize = widgets.Button(description='Optimize',layout=bl)
	vbox_align = widgets.VBox([dropdown_results,accordion_initializetheta,range_minmaxiter,dropdown_order,dropdown_downscale,button_optimize])

	## Plot widgets
	dropdown_zoom = widgets.Dropdown(value='Full', options=['Full','Center','TL','TR','BL','BR'],description='Plot Zoom',style=ws)
	button_plot = widgets.Button(description='Plot',layout=bl)
	vbox_plot = widgets.VBox([dropdown_zoom,button_plot,])

	## All together
	tabs_total = widgets.Tab(children=[vbox_prepare, vbox_align, vbox_plot,],titles=['Load','Align','Plot'])

	with out:
		display(tabs_total)

	##### Prepare
	def click_prepare(b):
		global fn_data,results_thetas,results_index,results_names,out_dir
		with out:
			out.clear_output()
			display(tabs_total)

			filename = re.sub(r'\s+', '', text_filename.value)
			 
			if os.path.exists(filename):
				print('Found: %s'%(filename))

				method = dropdown_method.value
				if method == 'Frame First': ## for now....
					method = int_start
				split = dropdown_split.value 
				first = int_start.value
				last = int_end.value
				aligner.prepare_data(filename,method,split,first,last) ## Note: this makes the results directory, if necessary
				
				fn_data = filename
				out_dir = aligner.get_out_dir(fn_data)

				## load any that were previously created
				fnthetas = [fn for fn in os.listdir(out_dir) if fn.endswith('.theta.npy')]
				if len(fnthetas) == 0:
					results_names.append('%04d_blank_order1_bin1'%(0))
					results_thetas.append(aligner.initialize_theta(fn_data,False,None,False))
					save_theta(results_names[0],results_thetas[0])
					results_index = 0
				else:
					indexes = [get_index(fn) for fn in fnthetas]
					order = np.argsort(indexes)
					for i in order:
						results_thetas.append(np.load(os.path.join(out_dir,fnthetas[i])))
						results_names.append(os.path.split(fnthetas[i])[1][:-10])
					results_index = len(results_names)-1
				dropdown_results.options = results_names[::-1]
				dropdown_results.value = results_names[results_index]

			else:
				print('Failure: File does not exist !!!! %s'%(filename))
				fn_data = None
				results_index = None
				results_names = []
				results_thetas = []
				out_dir = None

	##### Align
	def get_index(s):
		return int(s.split('_')[0])
	
	def save_theta(name,theta):
		global out_dir
		fn_out = os.path.join(out_dir,name+'.theta.npy')
		np.save(fn_out,theta)
	
	def get_last_index(thetalist):
		max_index = np.max([get_index(tli) for tli in thetalist])
		return max_index

	def add_new_theta(new_name,new_theta):
		results_names.append(new_name)
		results_thetas.append(new_theta)
		results_index = len(results_names)-1
		save_theta(results_names[results_index],results_thetas[results_index])
		update_resultsdropdown()
		dropdown_results.value = new_name

	def on_combo_box_change(change):
		global results_index
		if change['type'] == 'change' and change['name'] == 'value':
			results_index = results_names.index(change['new'])

	def update_resultsdropdown():
		global results_names
		dropdown_results.options = results_names[::-1]

	def click_loadtheta(b):
		global fn_data,results_thetas,results_index,results_names,out_dir
		with out:
			filename = re.sub(r'\s+', '', text_thetafilename.value)
			if os.path.exists(filename):
				new_theta = aligner.initialize_theta(fn_data,True,filename,False)
				order = aligner.alignment.coefficients_order(new_theta)
				new_name = '%04d_loaded_order%d_binX'%(get_last_index(results_names)+1,order)
				add_new_theta(new_name,new_theta)
			else:
				print('Failure: File does not exist !!!! %s'%(filename))

	def click_blanktheta(b):
		global fn_data,results_thetas,results_index,results_names,out_dir
		with out:
			new_name = '%04d_blank_order%d_bin%d'%(get_last_index(results_names)+1,1,1)
			new_theta = aligner.initialize_theta(fn_data,False,None,False)
			add_new_theta(new_name,new_theta)

	def click_fouriertheta(b):
		with out:
			new_name = '%04d_fourier_order%d_bin%d'%(get_last_index(results_names)+1,1,1)
			new_theta = aligner.initialize_theta(fn_data,False,None,True)
			add_new_theta(new_name,new_theta)

	def click_optimize(b):
		with out:
			out.clear_output()
			display(tabs_total)

			order = int(dropdown_order.value)
			downscale = float(dropdown_downscale.value)
			miniter,maxiter = range_minmaxiter.value
			old_theta = results_thetas[results_index]

			new_theta = aligner.optimize_data(fn_data,old_theta,float(downscale),int(order),True,maxiter,miniter)
			new_name = '%04d_optimize_order%d_bin%d'%(get_last_index(results_names)+1,order,downscale)
			add_new_theta(new_name,new_theta)

	#### Plots
	def click_plot(b):
		global fig,ax
		with out:
			out.clear_output()
			display(tabs_total)
			fig,ax = plot()
			plt.show()
	
	def plot():
		global fn_data,results_thetas,results_index,results_names,out_dir
		global fig,ax
		with out:
			out_dir = aligner.get_out_dir(fn_data)
			fig,ax = aligner.render_images(fn_data,results_thetas[results_index])
			[aa.axis('off') for aa in ax]
			fig.set_figheight(8.)
			fig.set_figwidth(20.)
			fig.tight_layout()
			plt.savefig(os.path.join(out_dir,'render_%s.png'%(results_names[results_index])))
			return fig,ax

	def change_zoom(change):
		if change['type'] == 'change' and change['name'] == 'value':
			global fig,ax
			with out:
				out.clear_output()
				display(tabs_total)
				ny,nx,_ = ax[0].images[0].get_array().shape
				ll = np.min((nx,ny))//4
				if dropdown_zoom.value == 'Full':
					ax[0].set_xlim(0,nx)
					ax[0].set_ylim(ny,0)
				elif dropdown_zoom.value == 'Center':
					ax[0].set_xlim(nx//2-ll,nx//2+ll)
					ax[0].set_ylim(ny//2+ll,ny//2-ll)
				elif dropdown_zoom.value == 'TL':
					ax[0].set_xlim(0,ll)
					ax[0].set_ylim(ll,0)
				elif dropdown_zoom.value == 'TR':
					ax[0].set_xlim(nx-ll,nx-1)
					ax[0].set_ylim(ll,0)
				elif dropdown_zoom.value == 'BL':
					ax[0].set_xlim(0,ll)
					ax[0].set_ylim(ny-1,ny-ll)
				elif dropdown_zoom.value == 'BR':
					ax[0].set_xlim(nx-ll,nx-1)
					ax[0].set_ylim(ny-1,ny-ll)
				fig.tight_layout()
				fig.canvas.draw()
				display(fig)

	button_prepare.on_click(click_prepare)

	dropdown_results.observe(on_combo_box_change)
	button_loadtheta.on_click(click_loadtheta)
	button_blanktheta.on_click(click_blanktheta)
	button_fouriertheta.on_click(click_fouriertheta)
	button_optimize.on_click(click_optimize)

	button_plot.on_click(click_plot)
	dropdown_zoom.observe(change_zoom)

	display(out)