import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import numba as nb

from scipy.special import erf as _erf
from scipy.special import gammaln as _gammaln
from scipy.special import betainc as _betainc

@nb.vectorize(cache=True)
def gammaln(q):
	with nb.objmode(y='float64'):
		y = _gammaln(q)
	return y

@nb.vectorize(cache=True)
def erf(q):
	with nb.objmode(y='float64'):
		y = _erf(q)
	return y

@nb.vectorize(cache=True)
def betainc(a, b, x):
	with nb.objmode(y='float64'):
		y = _betainc(a, b, x)
	return y

