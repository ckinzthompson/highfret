import multiprocessing
multiprocessing.set_start_method('forkserver', force=True) ## do this first or pyinstaller version will keep opening forever....
multiprocessing.freeze_support()
from flask import Flask, render_template, request, jsonify, send_file
import webview
import os
import threading
from functools import wraps
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_pdf ## to include in pyinstaller
import highfret

window = None
app = Flask(__name__)

def get_output_images(odir: Path):
	"""Find all PNG and PDF files in the output folder and pair them"""
	file_pairs = []  # List of {png: path, pdf: path, name: basename}
	
	odir = Path(odir)
	if not odir.exists():
		return file_pairs
	
	for fn in odir.iterdir():
		if fn.suffix == '.pdf':
			fpdf = fn.absolute()
			fpng = fn.parent / f'{fpdf.stem}.png'
			if fpng.exists():
				file_pairs.append({'name': str(fn.stem+fn.suffix), 'pdf':str(fpdf), 'png':str(fpng)})
	return file_pairs

def get_folder(fn: Path):
	fn = Path(fn).absolute()
	if not fn.exists():
		return False,''
	if not fn.suffix == '.tif':
		return False,''
	folder = str(highfret.containers.tif_folder.gen_folder_name(fn))
	return True,folder

def safely(func):
	@wraps(func)
	def wrapper(filename, *args, **kwargs):
		try:
			result = func(filename, *args, **kwargs)
			if isinstance(result, dict):
				return {"success": True, **result}
			return {
				"success": True,
				"result": "OK",
				**result
			}
		except Exception as e:
			return {"success": False, "error": str(e)}
	return wrapper

@safely
def refresh(filename,params):
	success,folder = get_folder(filename)
	if not success:
		return {"success": False, "error": f'No highFRET folder for {filename}'}
	images = get_output_images(folder)
	return {"success": True, "result": 'Refreshed', "images": images, "output_folder": folder}

@safely
def new(filename,params):
	highfret.core.new(filename,params['split'],params['bin'])
	success,folder = get_folder(filename)
	images = get_output_images(folder)
	return {"success": True, "result": 'New analysis created', "images": images, "output_folder": folder}

@safely
def intensity(filename,params):
	highfret.core.intensity(filename)
	success,folder = get_folder(filename)
	images = get_output_images(folder)
	return {"success": True, "result": 'Intensity plot made', "images": images, "output_folder": folder}

@safely
def align(filename,params):
	end = 0
	highfret.core.align(filename,params['start'],end,params['second'])
	success,folder = get_folder(filename)
	images = get_output_images(folder)
	return {"success": True, "result": 'Alignment completed', "images": images, "output_folder": folder}

@safely
def spotfind(filename,params):
	end = 0
	median = 21
	highfret.core.spotfind(filename,params['start'],end,params['which'],params['cutoff'],median)
	success,folder = get_folder(filename)
	images = get_output_images(folder)
	return {"success": True, "result": 'Spots found', "images": images, "output_folder": folder}

@safely
def extract(filename,params):
	dl=5
	max_restarts=15
	correct=False
	highfret.core.extract(filename,params['sigma'],dl,max_restarts,correct,params['fast'])
	success,folder = get_folder(filename)
	images = get_output_images(folder)
	return {"success": True, "result": 'Traces extracted', "images": images, "output_folder": folder}

@safely
def optimizepsf(filename,params):
	highfret.core.optimizepsf(filename)
	success,folder = get_folder(filename)
	images = get_output_images(folder)
	return {"success": True, "result": 'PSF optimized', "images": images, "output_folder": folder}

# @safely
# def auto(filename,params):
# 	highfret.core.auto(filename,params['start'])
# 	success,folder = get_folder(filename)
# 	images = get_output_images(folder)
# 	return {"success": True, "result": 'Automatic run complete', "images": images, "output_folder": folder}

class Api: ### API for PyWebView to expose to JavaScript
	def select_file(self):
		global window
		"""Open native file dialog and return full file path"""
		file_types = ('All Files (*.*)', 'Text Files (*.txt)', 'CSV Files (*.csv)')
		result = window.create_file_dialog(webview.FileDialog.OPEN, file_types=file_types)
		return result[0] if result else None
	
	def open_file(self, filepath):
		"""Open a file with the default system application"""
		import subprocess
		import platform
		try:
			if platform.system() == 'Windows':
				os.startfile(filepath)
			elif platform.system() == 'Darwin':  # macOS
				subprocess.run(['open', filepath])
			else:  # Linux
				subprocess.run(['xdg-open', filepath])
		except Exception as e:
			print(f"Error opening file: {e}")
	
	def open_folder(self, folderpath):
		"""Open a folder in the system file explorer"""
		import subprocess
		import platform
		try:
			if platform.system() == 'Windows':
				os.startfile(folderpath)
			elif platform.system() == 'Darwin':  # macOS
				subprocess.run(['open', folderpath])
			else:  # Linux
				subprocess.run(['xdg-open', folderpath])
		except Exception as e:
			print(f"Error opening folder: {e}")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/get_image/<path:filepath>')
def get_image(filepath):
	"""Serve images from the output folder"""
	try:
		if not filepath.startswith('/'):
			filepath = '/' + filepath
		if not os.path.exists(filepath):
			return f"File not found: {filepath}", 404
		response = send_file(filepath, mimetype='image/png' if filepath.endswith('.png') else 'application/pdf')
		return response
	except Exception as e:
		print(f"[get_image] EXCEPTION: {str(e)}")
		import traceback
		traceback.print_exc()
		return f"Error: {str(e)}", 404

@app.route('/run_method', methods=['POST'])
def run_method():
	data = request.json
	method = data.get('method')
	filename = data.get('filename')
	params = data.get('params')
	for key in params.keys():
		if params[key] == 'true':
			params[key] = True
		elif params[key] == 'false':
			params[key] = False
	if params['split'] == 'lr':
		params['split'] = 'l/r'
	if params['split'] == 'tb':
		params['split'] = 't/b'
	
	if not filename:
		return jsonify({"success": False, "error": "Please provide a filename"})
	
	methods = {
		'refresh': refresh, 
		'new':new,
		'intensity':intensity,
		'align':align,
		'spotfind':spotfind,
		'extract':extract,
		'optimizepsf':optimizepsf,
		# 'auto':auto,
	}
	if not method in methods:
		return jsonify({"success": False, "error": "Invalid method"})
	result = methods[method](filename,params)
	return jsonify(result)

def start_flask():
	"""Start Flask in a separate thread"""
	app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

def main():
	global window
	flask_thread = threading.Thread(target=start_flask, daemon=True)
	flask_thread.start()
	
	api = Api()
	window = webview.create_window(
		f'highFRET - {highfret.__version__}',
		'http://127.0.0.1:5000',
		js_api=api,
		width=1000,
		height=800,
		resizable=True,
		background_color='#FFFFFF'
	)
	
	webview.start()
	
if __name__ == '__main__':
	main()

# pyinstaller --windowed --name "highFRET" --add-data "templates:templates" webapp.py