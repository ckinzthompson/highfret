import shutil
import typer
from pathlib import Path
from typing_extensions import Annotated
from highfret import core
import matplotlib.pyplot as plt

cli_app = typer.Typer(help="highFRET: Analyze TIRF Movies for smFRET Experiments",no_args_is_help=True,add_completion=False)

@cli_app.command(help="Launch the standalone app")
def app():
	import multiprocessing
	multiprocessing.set_start_method('forkserver', force=True) ## do this first or pyinstaller version will keep opening forever....
	multiprocessing.freeze_support()
	from highfret.web.webapp import main
	main()
	typer.echo('Launched app')

@cli_app.command(help="New/clear analysis")
def new(
		movie_path: Annotated[Path,typer.Argument(help="Location of tif file movie to analyze")],
		split: Annotated[str,typer.Option(help="How color channels are arranged [none, l/r, t/b, quad]", rich_help_panel="Options")] = 'l/r',
		bin: Annotated[int,typer.Option(help="Binning (1,2,3,4,...)", rich_help_panel="Options")] = 1,
	):
	core.new(movie_path,split,bin)
	typer.echo('New analysis completed')

@cli_app.command(help="Plot of average pixel intensity in each color channel")
def intensity(
		movie_path: Annotated[Path,typer.Argument(help="Location of tif file movie to analyze")],
	):
	core.intensity(movie_path)
	typer.echo('Showing average pixel intensity plot')
	plt.show()

@cli_app.command(help="Aligns color channels")
def align(
		movie_path: Annotated[Path,typer.Argument(help="Location of tif file movie to analyze")],
		start: Annotated[int,typer.Option(help="Consider only frames from this point on", rich_help_panel="Options",min=0)] = 0,
		end: Annotated[int,typer.Option(help="Consider only frames before this point. 0 means the end", rich_help_panel="Options",min=0)] = 0,
		second: Annotated[bool, typer.Option("--second",help="Align up to second order", rich_help_panel="Options")] = False,
	):
	core.align(movie_path,start,end,second)
	typer.echo('Color channels aligned')
	plt.show()

@cli_app.command(help="Copy alignment from another movie")
def copy(
		from_path: Annotated[Path, typer.Argument(help="Location of tif file movie to analyze")],
		to_path: Annotated[Path, typer.Argument(help="Location of tif file movie to analyze")],
	):
	core.copy(from_path,to_path)
	typer.echo('Copied alignment')

@cli_app.command(help="Find spots in the molecule")
def spotfind(
		movie_path: Annotated[Path, typer.Argument(help="Location of tif file movie to analyze")],
		start: Annotated[int, typer.Option(help="Consider only frames from this point on", rich_help_panel="Options",min=0)] = 0,
		end: Annotated[int, typer.Option(help="Consider only frames before this point. 0 means the end", rich_help_panel="Options",min=0)] = 0,
		which: Annotated[str, typer.Option(help="Spots from which color channel? [all, green, red]", rich_help_panel="Options")] = 'all',
		cutoff: Annotated[float, typer.Option(help="Threshold value for maxima to be considered a spot. Value should be between (0:1)", rich_help_panel="Options")] = 0.15,
		median: Annotated[int, typer.Option(help="Median filter width to remove background from ACF image", rich_help_panel="Options")] = 21,
	):
	core.spotfind(movie_path,start,end,which,cutoff,median)
	typer.echo(f'Spot finding completed')
	plt.show()

	
@cli_app.command(help="Find the PSF of the spots")
def optimizepsf(	
		movie_path: Annotated[Path, typer.Argument(help="Location of tif file movie to analyze")],
	):
	core.optimizepsf(movie_path)
	typer.echo(f'Optimized PSF widths')
	plt.show()

@cli_app.command(help="Find spots in the molecule")
def extract(
		movie_path: Annotated[Path, typer.Argument(help="Location of tif file movie to analyze")],
		sigma: Annotated[float, typer.Option(help="PSF width (~HWHM) in pixels", rich_help_panel="Options")] = 0.8,
		dl: Annotated[int, typer.Option(help="Spot search radius in pixels (area=(2*dl+1)^2)", rich_help_panel="Options")] = 5,
		max_restarts: Annotated[int, typer.Option(help="Number of restarts for MLE", rich_help_panel="Options")] = 15,
		correct: Annotated[bool, typer.Option("--correct",help="Do an ad hoc baseline correction?", rich_help_panel="Options")] = False,
		fast: Annotated[bool, typer.Option("--fast",help="No MLE, just vanilla extraction?", rich_help_panel="Options")] = False,
	):
	core.extract(movie_path,sigma,dl,max_restarts,correct,fast)
	typer.echo(f'Extracted traces')
	plt.show()

@cli_app.command(help="Check the log")
def log(movie_path: Annotated[Path, typer.Argument(help="Location of tif file movie to analyze")],):
	core.log(movie_path)

@cli_app.command(help="Automatically do a full processing")
def auto(
		movie_path: Annotated[Path, typer.Argument(help="Location of tif file movie to analyze")],
		start: Annotated[int,typer.Option(help="Consider only frames from this point on", rich_help_panel="Options",min=0)] = 0,
	):
	core.auto(movie_path,start)
	typer.echo(f'Finished auto processing')
	plt.show()

if __name__ == "__main__":
	cli_app()