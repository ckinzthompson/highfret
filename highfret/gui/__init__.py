from .aligner_gui import gui_aligner as aligner
from .extracter_gui import gui_extracter as extracter
from .spotfinder_gui import gui_spotfinder as spotfinder


def highfret(fn_data='',fn_align='',fn_cal=''):
	# aligner(fn_data,)
	spotfinder(fn_data,fn_align,fn_cal)
	extracter(fn_data,fn_align,fn_cal)
