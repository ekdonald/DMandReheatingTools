###################################################################################################
###################################################################################################
#                                                                                                 #
#                               Primordial Black Hole Evaporation                                 #
#                         Dark Matter Production from Hawking Radiation                           #
#                                 Considering Mass Distributions                                  #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                                   Based on: arXiv:2212.XXXXX                                    #
#                                                                                                 #
###################################################################################################

#======================================================#
#                                                      #
#                     Example script                   #  
#                                                      #
#======================================================#

import sys
import numpy as np
from odeintw import odeintw
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta
from scipy.special import kn
import mpmath
from mpmath import polylog
import BHProp as bh
from SolFBEqs_Mono import FBEqs_Sol                   # Monochromatic Scenario
from SolFBEqs_MassDist import FBEqs_Sol as FBSol_MD   # Mass Distributions
from SolFBEqs_MassDist import Omega_h2DM as Omegah2   # Mass Distributions
import time
from datetime import datetime

# by DK
import Functions_phik as phik_functions
#import reheat_vars as rh_vars
# to make it possible to give arguments..######
import sys
import os


print('-------------------------------------')
print('-   Starting the run')
start = time.time()
print("Started at = ", datetime.now())
print('-------------------------------------')


#----------------------------------------#
#           Main Parameters              #
#----------------------------------------#

# Read reheating parameters
rhprocess = phik_functions.phik_process().process 
kvar = phik_functions.phik_process().kvar 
mueff = phik_functions.phik_process().mueff 
sigmaeff = phik_functions.phik_process().sigmaeff 

print('--------------------  Number of arguments:', len(sys.argv), 'arguments -------------------')

mDM     = float(sys.argv[1])
beta    = float(sys.argv[2])

## chose here the initial mass of the black hole
LogMmin      = float(sys.argv[3])  # free parameter
Delta_LogM  = float(sys.argv[4])
alpha   = float(sys.argv[5])
tag     = sys.argv[6]
yeff    = float(sys.argv[7])

print("alpha = ", alpha)
print("LogMmin, Mmin (g) = ", LogMmin, 10**LogMmin)


Mi   = LogMmin  # Peak mass in g at formation  --> Taken here as a parameter
asi  = 0.       # PBH initial rotation a_star factor
bi   = beta     # Initial PBH fraction 
mDM  = mDM      # Log10 @ Dark Matter mass
sDM  = 0.       # Dark Mater spin

##########################################
#  Distribution types:
#    0 -> Lognormal, requires 1 parameter, sigma
#    1 -> Power law, requires 2 parameters, [sigma, alpha]
#    2 -> Critical collapse, doesn't require any parameters
#    3 -> Metric preheating, doesn't require any parameters
#########################################

typ = 1 
sig = Delta_LogM 

print()
print("#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#")
print()
if typ == 0:
   print("#        Using Log-Normal distribution for the BH mass        #")
elif typ == 1:
   print("#        Using Power-Law distribution for the BH mass         #")
elif typ == 2:
   print("#    Using Critical collapse distribution for the BH mass     #")
elif typ == 3:
   print("#    Using Metric preheating distribution for the BH mass     #")
else:
   print("# Unkown distribution. Using Power-Law distribution as default #")
   typ = 1
print()
print("#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#")
print()

if typ == 1:
    pars_MD = [sig, alpha]
else:
    pars_MD = sig

#------------------------------------------------------------------------------------------------------#
#          We call the solver, and save the arrays containing the full evolution of the PBH,           #
#    SM and DR comoving energy densities, together with the evolution of the PBH mass and spin         #
#                              as function of the log_10 @ scale factor.                               #
#                  We compute for both monochromatic and mass distribution scenario                    #

#------------------------------------------------------------------------------------------------------#
print('mDM = 10^', mDM,', beta = 10^', beta,', Mmin = 10^', LogMmin,', Delta_Log_M = ', Delta_LogM)
print('--------------------------------------------------------------------------')
print('--- Distribution run ---')


if tag == "mono": 
#+++++++++++++++++++++++++++++#
#        Monochromatic        #
#+++++++++++++++++++++++++++++#
	start = time.time()
	Oh2m = FBEqs_Sol(Mi, asi, bi, mDM, sDM)
	xm, tm, MBHm, astm, Phim, Radm, PBHm, TUnm, NDBEm, Tevm  = Oh2m.Solt()
	end = time.time()
	print(f"\n Monochromatic Time {end - start} s\n") #

elif tag == "ext":
	#+++++++++++++++++++++++++++++#
	#       Mass Distribution     #
	#+++++++++++++++++++++++++++++#
	start = time.time()
	Oh2 = FBSol_MD(Mi, bi, typ, pars_MD, mDM, sDM)
	x, t, Phi, Rad, PBH, TUn, NDBE, Tev  = Oh2.Solt()
	end = time.time()
	print(f"Mass Distribution Time {end - start} s\n")

else: 
	print("unknown mass distribution. Using the extended dist.")
	tag = "ext"
	start = time.time()
	Oh2 = FBSol_MD(Mi, bi, typ, pars_MD, mDM, sDM)
	x, t, Phi, Rad, PBH, TUn, NDBE, Tev  = Oh2.Solt()
	end = time.time()
	print(f"Mass Distribution Time {end - start} s\n")



#---------------- PRINT OUT DATA -----------------------# 

print("WRITING OUT DATA FOR k = {}".format(int(kvar)))


tag_data = '_logbeta_' + str(int(beta)) + '_logMc_' +str(int(LogMmin)) + \
		   '_sigM_' +str(int(Delta_LogM))+ '_siga_' + '_alpha_' +str(int(alpha))

pre_name = 'Results/k='+str(int(kvar))+'/data_scan_'
  

saving_name = pre_name + tag + '_log10a' + tag_data + '.txt'
np.savetxt(saving_name, x, delimiter=',')

saving_name = pre_name + tag + '_a4rhorad' + tag_data + '.txt'
np.savetxt(saving_name, Rad, delimiter=',')

saving_name = pre_name + tag + '_funkrhophi' + tag_data + '.txt'
np.savetxt(saving_name, Phi, delimiter=',')

saving_name = pre_name + tag + '_a3rhoPBH' + tag_data + '.txt'
np.savetxt(saving_name, PBH, delimiter=',')

saving_name = pre_name + tag + '_TPBH' + tag_data + '.txt'
np.savetxt(saving_name, TUn, delimiter=',')

saving_name = pre_name + tag + '_a3nDM' + tag_data + '.txt'
np.savetxt(saving_name, NDBE, delimiter=',')


incols = ["logbeta","logMc","sigma_M","alpha","rhprocess","kvar","yeff","mueff","sigmaeff", "tag", "logmDM", "Tev"]
fd = pd.DataFrame(columns=incols)
listo = [beta, LogMmin, Delta_LogM, -alpha, rhprocess, kvar, yeff, mueff, sigmaeff, tag, mDM, Tev]
fd.loc[len(fd)] = listo
fd.to_csv(pre_name+"inparameters.dat", header=None, index=None, sep=' ', mode='w+')    



#------------------------------------------------------------#
#                                                            #
#                     Determining Oh^2                       #
#                                                            #
#------------------------------------------------------------#

Oh2 = Omegah2()
ff = open("Omegah2.dat", "w")
ff.write(str(Oh2))
ff.close()


print("#--------------------------------#")
print(" ")
print("Omega h^2 for {}".format(Oh2))
print(" ")
print("#--------------------------------#")
#'''


end = time.time()
print('-------------------------------------')
print("Ended at = ", datetime.now())
print('The Full run took ', end-start, ' seconds.')
print('-------------------------------------')
