###################################################################################################
#                                                                                                 #
#                         Primordial Black Hole + Dark Matter Generation.                         #
#                                     Evaporation + Freeze-In                                     #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                    Based on: arXiv:2107.00013 (P1) and  arXiv:2107.00016 (P2)                   #
#                                                                                                 #
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.misc import derivative
from matplotlib import gridspec
from matplotlib import rc
import pandas as pd
import seaborn as sb
import matplotlib as mpl
import scipy as sp
import matplotlib.colors as colors
import math
import os
import subprocess
import matplotlib.ticker as ticker
from matplotlib import ticker as mticker
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
from odeintw import odeintw
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, quad_vec, ode, solve_ivp, odeint, romberg, dblquad, fixed_quad
from sympy.solvers import solve
from sympy import Symbol 
from matplotlib.colors import Normalize
from IPython.display import display, Latex
import Functions_phik as funcs_phik
from scipy.special import zeta, kn, spherical_jn, jv
import BHProp as bh #Schwarzschild and Kerr BHs library
#from num2tex import num2tex
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,bm}')
f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = ticker.FuncFormatter(g)
from IPython.display import display, Latex



# Constants

c     = 299792.458       # in km/s
gamma = np.sqrt(3.)**-3.    # Collapse factor
GCF   = 6.70883e-39      # Gravitational constant in GeV^-2
mPL   = GCF**-0.5        # Planck mass in GeV
v     = 174              # Higgs vev
csp   = 0.35443          # sphaleron conversion factor
GF    = 1.1663787e-5     # Fermi constant in GeV^-2

## Define parameters
Mp_real = 1.22e19
Mp = 2.435e18 
geff = 100 # 427/4
alpha_0 = 1   
alpha_1 = np.sqrt(2/(3*alpha_0))
AR_CMB = 2.19e-09
ns_CMB = 0.9645
alphaStar = geff*np.pi**2/30
E_BBN = 5.0e-03

GeV_in_g     = 1.782661907e-24  # 1 GeV in g
Mpc_in_cm    = 3.085677581e24   # 1 Mpc in cm
cm_in_invkeV = 5.067730938543699e7       # 1 cm in keV^-1
year_in_s    = 3.168808781402895e-8      # 1 year in s
GeV_in_invs  = cm_in_invkeV * c * 1.e11  # 1 GeV in s^-1
MPL   = mPL * GeV_in_g        # Planck mass in g
kappa = mPL**4 * GeV_in_g**3  # Evaporation constant in g^3 * GeV -- from PRD41(1990)3052



#----------------------------------------------------------------------#
#----------------             Omega_h2DM              -----------------#
#----------------------------------------------------------------------#

def Omega_h2DM(x, TUn, NDMH, logmDM, aev, aRH, TRH):
    '''
    This function directly returns Omega_h2 for DM produced by PBH evaporation
    '''
    mDM = 10.**logmDM
    T0 = 2.34865e-13                               # Temperature today in GeV
    nphi = (2.*zeta(3)/np.pi**2)*TUn[0]**3         # Initial photon number density
    rc = 1.053672e-05*cm_in_invkeV**-3*1.e-18      # Critical density in GeV^4
    
    inv_rhocrit = 1./rc
    gS_ratio = (bh.gstarS(T0)/bh.gstarS(TUn.iloc[-1]))
    T_ratio = (T0/TRH)**3    
    a_ratio = (aev/aRH)**3

    print("n_DM(aev) = ", nphi*NDMH.iloc[-1]*10.**(-3.*x.iloc[-1]))
    
    if (aev <= aRH): 
        #'''
        print("Evaporation before reheating")
        #'''
        Oh2  = (nphi*NDMH.iloc[-1])*10.**(-3.*x.iloc[-1])*mDM*(gS_ratio)*(T_ratio)*(a_ratio)*inv_rhocrit
    else:
        #'''
        print("Evaporation after reheating")
        #'''
        Oh2  = (nphi*NDMH.iloc[-1])*10.**(-3.*x.iloc[-1])*mDM*(gS_ratio)*(T_ratio)*inv_rhocrit
    return Oh2


