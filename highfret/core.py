import shutil
from . import container,containers,prepare,aligner,spotfinder,extracter

def new(movie_path,split='l/r',bin=1):
	analysis = container.new(movie_path,split,None,bin)
	analysis.save()

def intensity(movie_path):
	analysis = container.load(movie_path)
	fig = prepare.plot_avg_intensity(analysis)
	[analysis.export('figure',fig=fig,fname=f'fig_pixel_avgintensity.{ext}') for ext in ['pdf','png']]

def align(movie_path,start=0,end=0,second=True):	
	analysis = container.load(movie_path)

	prepare.prepare_img(analysis,start,end)
	aligner.clear(analysis)
	aligner.initialize(analysis)
	aligner.optimize(analysis,1,1.,10,5)
	if second:
		aligner.optimize(analysis,2,1.,10,5)
	analysis.save()
	figaxs = aligner.render(analysis)
	for i in range(len(figaxs)):
		[analysis.export('figure',fig=figaxs[i][0],fname=f'fig_align_{i}.{ext}') for ext in ['pdf','png']]

def copy(from_path,to_path):
	folder = containers.tif_folder.gen_folder_name(from_path)
	fn = folder / 'transforms.npy'
	if not fn.exists():
		print('Movie has no alignment to copy: {fn.stem}')
		return

	folder = containers.tif_folder.gen_folder_name(to_path)
	if not folder.exists():
		folder.mkdir()
	shutil.copy(fn,folder / 'transforms.npy')

def spotfind(movie_path,start=0,end=0,which='all',cutoff=0.15,median=21):
	analysis = container.load(movie_path)

	matched = True
	if not which in ['all','green','red']:
		which = 'all'
	elif which == 'green':
		which = 'only 0'
		matched = False
	elif which == 'red':
		which = 'only 1'
		matched = False

	prepare.prepare_img(analysis,start,end,median)
	fig,ax = spotfinder.render_overlay(analysis,0.5)
	fig1,fig2 = spotfinder.run_spotfind(analysis, matched_spots=matched, which=which, acf_cutoff=cutoff)
	analysis.save()
	[analysis.export('figure',fig=fig,fname=f'fig_spotfind_overlay.{ext}') for ext in ['pdf','png']]
	[analysis.export('figure',fig=fig1,fname=f'fig_spotfind_separate.{ext}') for ext in ['pdf','png']]
	[analysis.export('figure',fig=fig2,fname=f'fig_spotfind_final.{ext}') for ext in ['pdf','png']]
	
def optimizepsf(movie_path):	
	analysis = container.load(movie_path)
	fig, ax, record_median, record_maximum = extracter.optimize_sigma(analysis)
	[analysis.export('figure',fig=fig,fname=f'fig_psf_optimize.{ext}') for ext in ['pdf','png']]

def extract(movie_path, sigma=0.8, dl=5, max_restarts=15,correct=False,fast=False):
	analysis = container.load(movie_path)
	if fast:
		extracter.extract_vanilla(analysis, sigma=sigma)
	else:
		extracter.extract_mle(analysis,sigma=sigma,max_restarts=max_restarts,correct=correct,dl=dl)
	analysis.save()

	fig,ax = extracter.figure_avg_intensity(analysis)
	# analysis.export('npy') ## done in analysis.save()...
	[analysis.export('figure',fig=fig,fname=f'fig_extracted_avgintensity.{ext}') for ext in ['pdf','png']]

def log(movie_path):
	analysis = container.load(movie_path)
	print(analysis.log)

def auto(movie_path,start=0):
	## Make a new analysis folder
	analysis = container.new(movie_path,'l/r',None,1)
	analysis.save()

	## Do the alignment
	prepare.prepare_img(analysis,start,0)
	aligner.initialize(analysis)
	aligner.optimize(analysis,1,1.,10,10)
	aligner.optimize(analysis,2,1.,10,10)
	
	## Find spots
	prepare.prepare_img(analysis,start,0,median=21)
	fig,ax = spotfinder.render_overlay(analysis,0.5)
	fig1,fig2 = spotfinder.run_spotfind(analysis)
	[analysis.export('figure',fig=fig, fname=f'fig_spotfind_overlay.{ext}') for ext in ['pdf','png']]
	[analysis.export('figure',fig=fig1,fname=f'fig_spotfind_separate.{ext}') for ext in ['pdf','png']]
	[analysis.export('figure',fig=fig2,fname=f'fig_spotfind_final.{ext}') for ext in ['pdf','png']]

	## Extract
	extracter.extract_mle(analysis,sigma=0.8,max_restarts=15,correct=False,dl=6)
	fig,ax = extracter.figure_avg_intensity(analysis)
	analysis.save()
	# analysis.export('npy') ## done in analysis.save()
	[analysis.export('figure',fig=fig,fname=f'fig_extracted_avgintensity.{ext}') for ext in ['pdf','png']]
