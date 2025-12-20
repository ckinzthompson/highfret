import os
import h5py
import time
import numpy as np
from pathlib import Path

from ..support import prepare
from ..support import modelselect_alignment as alignment

from .general import general_analysis_class	

def gen_folder_name(fn_data):
	if str(fn_data).endswith('.ome.tif'):
		folder = fn_data.parent / f"highfret_{fn_data.stem[:-4]}"
	else:
		folder = fn_data.parent / f"highfret_{fn_data.stem}"
	return folder

class analysis_tif_folder(general_analysis_class):
	def __init__(self):
		self._data = None
		self._img= None
		self._transforms = None
		self._spots = None
		self._traces = None
		self._log = None

		self.split = None
		self.ncolors = None
		self.fn_data = None
		self.folder = None

	@classmethod
	def new(self, fn_data:Path, split:str='l/r', folder=None, bin:int=1, verbose=True):
		fn_data = Path(fn_data)
		assert fn_data.exists()
		assert split.lower() in ['none','l/r','t/b','quad']
		assert bin > 0
		
		analysis = analysis_tif_folder()
		analysis.fn_data = Path(fn_data)
	
		if folder is None:
			folder = gen_folder_name(analysis.fn_data)
		analysis.folder = Path(folder)
		analysis.folder.mkdir(exist_ok=True)
		for fn in analysis.folder.iterdir():
			if fn.stem.startswith('fig_'):
				fn.unlink()
			elif fn.suffix == '.npy':
				fn.unlink()

		analysis.split = split.lower()
		analysis.ncolors = {'none':1,'l/r':2,'t/b':2,'quad':4}[analysis.split]
		analysis.bin = bin
		analysis.verbose = verbose
		
		analysis._data = None
		analysis._img = None
		_base_transform = alignment.coefficients_blank(1)
		analysis._transforms = np.array([[_base_transform for j in range(analysis.ncolors)] for i in range(analysis.ncolors)])
		analysis._spots  = np.zeros((0,2))
		analysis._traces  = np.zeros((0,0,analysis.ncolors))
		analysis._log = f'highFRET analysis - {time.ctime()}\n\tTIF File Path:{analysis.fn_data}\n\tOutput Folder Path:{analysis.folder}\n\tSplit Direction:{analysis.split}\n'
		return analysis

	@classmethod
	def load(self, fn_data:Path, folder=None,):
		fn_data = Path(fn_data)
		assert fn_data.exists()
		if folder is None:
			folder = gen_folder_name(fn_data)

		if not folder.exists():
			raise Exception(f'No analysis stored in :{folder}')
		
		analysis = analysis_tif_folder()
		analysis.fn_data = Path(fn_data)
		analysis.folder = Path(folder)
		
		## They might not all exist yet
		if (analysis.folder / 'img.npy').exists():
			analysis.img = np.load(analysis.folder / 'img.npy')
		if (analysis.folder / 'transforms.npy').exists():
			analysis.transforms = np.load(analysis.folder / 'transforms.npy')
		if (analysis.folder / 'spots.npy').exists():
			analysis.spots = np.load(analysis.folder / 'spots.npy')
		if (analysis.folder / 'traces.npy').exists():
			analysis.traces = np.load(analysis.folder / 'traces.npy')
		if (analysis.folder / 'log.txt').exists():
			with (analysis.folder / 'log.txt').open('r') as f:
				analysis.log = f.read()
		if (analysis.folder / 'config.txt').exists():
			with (analysis.folder / 'config.txt').open('r') as f:
				analysis.split = str(f.readline()[:-1].lower())
				analysis.bin = int(f.readline()[:-1])
				analysis.verbose = bool(str(f.readline()[:-1])=='True')
				analysis.ncolors = {'none':1,'l/r':2,'t/b':2,'quad':4}[analysis.split]
		return analysis

	def save(self):
		if not self._img is None:
			np.save(self.folder / 'img.npy', self.img)
		np.save(self.folder / 'transforms.npy', self.transforms)
		np.save(self.folder / 'spots.npy', self.spots)
		np.save(self.folder / 'traces.npy', self.traces)
		with (self.folder / 'log.txt').open('w') as f:
			f.write(self.log)
		with (self.folder / 'config.txt').open('w') as f:
			f.write(f'{self.split}\n{str(self.bin)}\n{str(self.verbose)}\n')

	def copy_alignment(self, fn:Path):
		fn = Path(fn)
		if fn.is_dir():
			fnn = fn / 'transforms.npy'
			if fnn.exists():
				self.transforms = np.load(fnn)
				return
			else:
				raise Exception(f'File does not exist: {fnn}')
		elif fn.stem == 'transforms' and fn.suffix == '.npy':
			self.transforms = np.load(fn)
			return
		elif fn.suffix == '.tif':
			folder = gen_folder_name(fn)
			if folder.exists():
				fnn = folder / 'transforms.npy'
				if fnn.exists():
					self.transforms = np.load(fnn)
					return
				else:
					raise Exception(f'File does not exist: {fnn}')
		raise Exception(f'Cannot load: {fn}')

	def make_data(self):
		self.log = self.log + f'Data IO:\n\tLoading: {self.fn_data}\n'
		t0 = time.time()
		self._data = prepare.load(self.fn_data,self.bin,self.verbose)
		self.log = self.log + f'\tBinned: {self.bin}x\n'
		t1 =time.time()

		self.log += f'\tTime Elapsed: {t1-t0:.3f}s\n'
		assert self._data.ndim == 3 ## T,X,Y
		if self.split == 'l/r':
			self._data = prepare.split_lr(self._data)
		elif self.ncolors == 't/b':
			self._data = prepare.split_tb(self._data)
		elif self.ncolors == 'quad':
			self._data = prepare.split_quad(self._data)
		assert self._data.ndim == 4 ## C,T,X,Y
		self.log += f'\tSplit Direction: {self.split}\n'

	## TODO: Fix this .... the issue is data is split into colors already...
	# def calibrate(self,calibration):
	# 	self.data.shape ## force-load
	# 	if calibration.size == 3: ## common for all pixels
	# 		cal = np.zeros((3,self.data.shape))
	# 	elif calibration.ndim == 3: ## per pixel

	# 		if fn_cal is None:
	# 	calibration = np.zeros((3,data.shape[1],data.shape[2])) ## _,o,v
	# 	calibration[0] += 1. ## g,_,_
	# else:
	# 	calibration = np.load(fn_cal) ## g,o,v	
	# print('Calibrating')
	# prepare.apply_calibration(data,calibration) ## preparation is in place
	
	@property
	def data(self): ## only load in if we have to
		if self._data is None:
			self.make_data()
		return self._data

	@property
	def log(self):
		return self._log
	@log.setter
	def log(self, value: str):
		self._log = value

	@property
	def img(self):
		if self._img is None:
			if self._data is None:
				self.make_data()
			ds = self._data.shape
			self._img = np.zeros((ds[0],ds[2],ds[3]))
		return self._img
	@img.setter
	def img(self, value: np.ndarray):
		self._img = value
	
	@property
	def transforms(self):
		return self._transforms
	@transforms.setter
	def transforms(self, value: np.ndarray):
		self._transforms = value

	@property
	def spots(self):
		return self._spots
	@spots.setter
	def spots(self, value: np.ndarray):
		self._spots = value

	@property
	def traces(self):
		return self._traces
	@traces.setter
	def traces(self, value: np.ndarray):
		self._traces = value
	
	def export(self, format='hdf5',*args,**kwargs):
		if format == 'hdf5':
			fn = self.folder / 'intensities.hdf5'
			self.log += f'Writing data to: {fn}\n'
			if fn.is_file():
				os.remove(fn)
			with h5py.File(fn,'w') as f:
				f.create_dataset('intensities', data=self.traces, dtype='float64',compression="gzip")
				f.flush()
				f.close()
		# elif format == 'npy':
		# 	fn = self.folder / 'intensities.npy'
		# 	self.log += f'Writing data to: {fn}\n'
		# 	np.save(fn,self.traces)
		elif format == 'figure':
			assert 'fig' in kwargs
			assert 'fname' in kwargs
			fig = kwargs['fig']
			fname = kwargs['fname']
			fig.savefig(self.folder / f'{fname}', dpi=300)
		else:
			raise Exception(f'Format {format} not yet implemented')
	
	
