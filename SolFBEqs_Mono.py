###################################################################################################
#                                                                                                 #
#                         Primordial Black Hole + Dark Matter Generation.                         #
#                                    Only DM from evaporation                                     #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                    Based on: arXiv:2107.00013 (P1) and  arXiv:2107.00016 (P2)                   #
#                                                                                                 #
###################################################################################################

import numpy as np
from odeintw import odeintw
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta, kn
from scipy.interpolate import interp1d, RectBivariateSpline

from math import sqrt, log, exp, log10, pi, atan

import BHProp as bh #Schwarzschild and Kerr BHs library

from collections import OrderedDict
olderr = np.seterr(all='ignore')

import time

import warnings
warnings.filterwarnings('ignore')

# by DK 05 June 2023
import Functions_phik as funcs_phik


# -------------------- Main Parameters ---------------------------
#
#
#          - 'Mi'   : Primordial BH initial Mass in grams
#
#          - 'ai'   : Primordial BH initial angular momentum a*
#
#          - 'bi'   : Primordial BH initial fraction beta^prim
#
#          - 'mDM'  : DM Mass in GeV
#
#          - 'sDM'  : DM spin -> [0.0, 0.5, 1.0, 2.0]
#
#          - 'g_DM' : DM degrees of freedom
#
#-----------------------------------------------------------------

#--------------------------   Credits  -----------------------------#
#
#      If using this code, please cite:
#
#      - arXiv:2107.00013,  arXiv:2107.00016                        #
#
#-------------------------------------------------------------------#

def StopMass(t, v, Mi):
    
    eps = 0.01
        
    if (eps*Mi > bh.MPL): Mst = eps*Mi
    else: Mst = bh.MPL

    return v[0] - Mst # Function to stop the solver if the BH is equal or smaller than the Planck mass

#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(x, v, nphi, mDM, sDM, Mi, xilog10):

    M     = v[0] # PBH mass
    ast   = v[1] # PBH ang mom
    rPhik = v[2] # Inflaton energy density in GeV^4 
    rRAD  = v[3] # Radiation energy density
    rPBH  = v[4] # PBH energy density
    Tp    = v[5] # Temperature
    NDMH  = v[6] # PBH-induced DM number density
    t     = v[7] # time in GeV^-1

    xff = (x + xilog10)

    #----------------#
    #   Parameters   #
    #----------------#

    M_GeV = M/bh.GeV_in_g     # PBH mass in GeV
    
    FSM = bh.fSM(M, ast)           # SM contribution
    FDM = bh.fDM(M, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM                # Total Energy contribution

    GSM = bh.gSM(M, ast)           # SM contribution
    GDM = bh.gDM(M, ast, mDM, sDM) # DM contribution
    GT  = GSM + GDM                # Total Angular Momentum contribution

    #-------- ADDED BY DONALD ON 15.11.2022 -------#
    #    Radiation density from inflaton decay     #
    #----------------------------------------------#
    process = funcs_phik.phik_process().process        		  
    k = funcs_phik.phik_process().kvar
    Mpla = funcs_phik.phik_process().Mp               		 
    lambdac = funcs_phik.phik_funcs(process).lambdavar(k)
	
    # Inflaton decay contribution
    fact_phi = 10.**((-6*k/(k+2))*xff)
    fact_rad = 10.**(-4*xff)
    gammaphi = funcs_phik.phik_funcs(process).gammaphi(k,lambdac)
    lpar = funcs_phik.phik_process().lparameter(k)
    omega_phi = funcs_phik.phik_funcs(process).omegaphi(k)
    Gamma_phi = gammaphi*np.power(fact_phi*rPhik/Mpla**4,lpar)
    
    Amaxin = funcs_phik.Amax_funcs().Amaxin 
    #Amaxin = funcs_phik.phik_funcs(process).Amaxin 
 	
    # contribution of inflaton to the radiation
    rRadphi = Gamma_phi*fact_phi*rPhik*(1.+omega_phi)

    #----------------#
    #   Parameters   #
    #----------------#
    
    #H   = sqrt(8 * pi * bh.GCF * (rPBH * 10.**(-3*x) + rRad * 10.**(-4*x))/3.) # Hubble parameter
    H = sqrt(8*pi*bh.GCF*(rPBH*10.**(-3*xff) + rPhik*10**(-(6*k/(k+2))*xff) + rRAD*10.**(-4*xff) )/3.) 
    #H   = np.sqrt(8*pi*bh.GCF*(rPBH * 10.**(-3*xff) + rRAD * 10.**(-4*xff))/3.) # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    dMdx     = - bh.kappa * FT/(M*M)/H
    dastdx   = - ast*bh.kappa*(GT - 2.*FT)/(M*M*M)/H
    drPhikdx = - rRadphi*10**((6*k/(k+2))*xff)/H 
    #drRADdx = - (FSM/FT)*(dMdx/M)*10.**xff*rPBH
    drRADdx  = - (FSM/FT)*(dMdx/M)*10.**xff*rPBH + rRadphi*10**(4*xff)/H 
    drPBHdx  = + (dMdx/M) * rPBH
    dTdx     = - (Tp/Del) * (1.0 - (bh.gstarS(Tp)/bh.gstar(Tp))*(0.25*drRADdx/rRAD))
    
    dtdx     = 1./H 

    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#
    
    dNDMHdx = (bh.Gamma_DM(M, ast, mDM, sDM)/H)*(rPBH/(M/bh.GeV_in_g))/nphi # PBH-induced contribution w/o contact
    
    ##########################################################    
    
    dEqsdx = [dMdx, dastdx, drPhikdx, drRADdx, drPBHdx, dTdx, dNDMHdx, dtdx]

    return [xeq * log(10.) for xeq in dEqsdx]


#----------------------------------#
#    Equations after evaporation   #
#----------------------------------#


def FBEqs_aBE(x, v):

    t     = v[0] # Time in GeV^-1
    rPhik = v[1] # Inflaton energy density in GeV^4  # DK
    rRAD  = v[2] # Radiation energy density
    Tp    = v[3] # Temperature
    NDMH  = v[4] # Thermal DM number density w/o PBH contribution
    
	#-------- ADDED BY DONALD ON 15.11.2022 -------#
	#    Radiation density from inflaton decay     #
	#----------------------------------------------#
    process = funcs_phik.phik_process().process        		  
    k = funcs_phik.phik_process().kvar
    Mpla = funcs_phik.phik_process().Mp               		 
    lambdac = funcs_phik.phik_funcs(process).lambdavar(k)
	
    # Inflaton decay contribution
    fact_phi = 10.**((-6*k/(k+2))*x)
    fact_rad = 10.**(-4*x)
    gammaphi = funcs_phik.phik_funcs(process).gammaphi(k,lambdac)
    lpar = funcs_phik.phik_process().lparameter(k)
    omega_phi = funcs_phik.phik_funcs(process).omegaphi(k)
    Gamma_phi = gammaphi*np.power(fact_phi*rPhik/Mpla**4,lpar)
    
    Amaxin = funcs_phik.Amax_funcs().Amaxin 
    #Amaxin = funcs_phik.phik_funcs(process).Amaxin 
 	
    # contribution of inflaton to the radiation
    rRadphi = Gamma_phi*fact_phi*rPhik*(1.+omega_phi)

    #----------------#
    #   Parameters   #
    #----------------#

    #H   = sqrt(8 * pi * bh.GCF * (rRad * 10.**(-4*x))/3.)    # Hubble parameter
    H = sqrt(8*pi*bh.GCF*(rPhik*10**(-(6*k/(k+2))*x) + rRAD*10.**(-4*x))/3.) 
    #H   = sqrt(8*pi*bh.GCF*(rRAD*10.**(-4*x))/3.)       # Hubble parameter
    Del = 1. + Tp*bh.dgstarSdT(Tp)/(3.0*bh.gstarS(Tp))     # Temperature parameter
    
    #----------------------------------------#
    #    Radiation + Temperature equations   #
    #----------------------------------------#

    dtdx    = 1./H
    drPhikdx = 0. - rRadphi*10**((6*k/(k+2))*x)/H  
    drRADdx  = 0. + rRadphi*10**(4*x)/H 
    #drRADdx = 0.
    dTdx    = - Tp/Del
        
    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#

    dNDMHdx = 0.                              # PBH-induced contribution w/o contact
        
    dEqsdx = [dtdx, drPhikdx, drRADdx, dTdx, dNDMHdx]

    return [xeq * log(10.) for xeq in dEqsdx]

#------------------------------------------------------------------------------------------------------------------#
#                                            Input parameters                                                      #
#------------------------------------------------------------------------------------------------------------------#
class FBEqs_Sol:
    ''' 
    Friedmann - Boltzmann equation solver for Primordial Black Holes + SM radiation + Dark Matter. See arXiv:2107.00013 2107.0001
    Monochromatic mass and spin scenario
    This class returns the full evolution of the PBH, SM and DR comoving energy densities,
    together with the evolution of the PBH mass and spin as function of the log_10 @ scale factor.
    '''
    
    def __init__(self, MPBHi, aPBHi, bPBHi, mDM, sDM):

        self.MPBHi  = MPBHi # Log10[M/1g]
        self.aPBHi  = aPBHi # a_star
        self.bPBHi  = bPBHi # Log10[beta']
        self.mDM    = mDM
        self.sDM    = sDM
    
#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                       Input parameters                                                              #
#-------------------------------------------------------------------------------------------------------------------------------------#
    
    def Solt(self):

        # Main parameters
        
        Mi     = 10**(self.MPBHi) # PBH initial Mass in grams
        asi    = self.aPBHi       # PBH initial rotation a_star factor
        bi     = 10**(self.bPBHi) # Initial PBH fraction

        # We assume a Radiation dominated Universe as initial conditions
        
        #Ti     = ((45./(16.*106.75*(pi*bh.GCF)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature
        #rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4                                          # Initial radiation energy density
        #rPBHi  = abs(bi/(sqrt(bh.gamma) -  bi))*rRadi                                         # Initial PBH energy density
        #nphi   = (2.*zeta(3)/pi**2)*Ti**3                                                     # Initial photon number density
        #ti = (np.sqrt(45./(16.*np.pi**3.*bh.gstar(Ti)*bh.GCF))*Ti**-2)                        # Initial time


        ##-------------------- DK
        process = funcs_phik.phik_process().process      
        k = funcs_phik.phik_process().kvar
        Mpla = funcs_phik.phik_process().Mp  
        
        Amaxin = funcs_phik.Amax_funcs().Amaxin              
        #Amaxin = funcs_phik.phik_funcs(process).Amaxin    

        lambdac = funcs_phik.phik_funcs(process).lambdavar(k)
        lpar = funcs_phik.phik_process().lparameter(k)
        Gklvar = funcs_phik.phik_funcs(process).Gkl(k, lambdac)
        rhoRadi = funcs_phik.phik_funcs(process).rhoRad(k, Amaxin)
        rhoend = funcs_phik.phik_funcs(process).rhoendvar(k)
        rhophi_ini = funcs_phik.phik_funcs(process).rhophi(k, Amaxin, rhoend) 

        Ti_PBH = ((45./(16.*106.75*(pi*bh.GCF)**3.))**0.25)*sqrt(bh.gamma*bh.GeV_in_g/Mi) 
        Ti     = (rhoRadi/(106.75*np.pi**2/30.0))**0.25
        rPBHi  = bi*(rhophi_ini + rhoRadi)                           # Initial PBH energy density
        nphi   = (2.*zeta(3)/pi**2)*Ti**3                            # Initial photon number density
        ti     = (sqrt(45./(16.*pi**3.*bh.gstar(Ti_PBH)*bh.GCF))*Ti_PBH**-2)  # Initial time, assuming a radiation dom Universe
        #ti    = (sqrt(45./(16.*pi**3.*bh.gstar(Ti)*bh.GCF))*Ti**-2)   # Initial time, assuming a radiation dom Universe

        NDMHi  = 0.0          # Initial DM comoving number density, in GeV^3     
        mDM  = 10**self.mDM   # DM mass in GeV
        sDM  = self.sDM       # DM spin

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                                           Solving the equations                                                   #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        xilog10 =  0.  #np.log10(Amaxin) 

        Min  = Mi
        asin = asi

        xBE    = []
        MBHBE  = []
        astBE  = []
        PhikBE = []
        RadBE  = []
        PBHBE  = []
        TBE    = []
        NDMHBE = []
        tmBE   = []

        MBHFn  = []
        NDMHFn = []
        
        taur = []

        i = 0
        
        while Mi >= 2. * bh.MPL:# Loop on the solver such that BH mass reaches M_Planck

            #--------------------------------------------------------------------------------#
            #         Computing PBH lifetime and scale factor in which BHs evaporate         #
            #--------------------------------------------------------------------------------#
            
            tau_sol = solve_ivp(fun=lambda t, y: bh.ItauFO(t, y, mDM, sDM), t_span = [-80, 40.], y0 = [Mi, asi], 
                                 rtol=1.e-5, atol=1.e-20, dense_output=True)
            
            if i == 0:
                #print("test i==0")
                Sol_t = tau_sol.sol   # Solutions for obtaining <p>
                tau = tau_sol.t[-1]   # Log10@PBH lifetime in inverse GeV
            
            if bi > 1.e-19*(1.e9/Mi):
                #print("bi > 1.e-19*(1.e9/Mi)")
                xf = root(bh.afin, [40.], args = (rPBHi, rhoRadi, 10.**tau, 0.), method='lm', tol=1.e-50) # Scale factor 
                xflog10 = xf.x[0]            
            else:
                #print("bi <= 1.e-19*(1.e9/Mi)")
                xfw = np.sqrt(1. + 4.*10.**tau*np.sqrt(2.*np.pi*bh.GCF*rhoRadi/3.))
                xflog10 = np.log10(xfw)
            
            #print("tau_sol.t[-1], tau = ", tau_sol.t[-1], tau)
            
            #-----------------------------------------#
            #          Before BH evaporation          #
            #-----------------------------------------#

            StopM = lambda t, x:StopMass(t, x, Mi) # Event to stop when the mass is 1% of the initial mass
            StopM.terminal  = True
            StopM.direction = -1.
            
            fact_nphi = np.power(Amaxin, 6*k/(k+2))
            rhophi_in = fact_nphi*rhophi_ini    

            fact_rPBHi = Amaxin**3
            rPBHi  = fact_rPBHi*rPBHi  

            fact_rRadi = Amaxin**4
            rRadi = fact_rRadi*rhoRadi


            v0 = [Mi, asi, rhophi_in, rRadi, rPBHi, Ti, NDMHi, ti]

            def stopphi(t, y):
                return y[2]*y[7]**(-(6*k/(k+2))) - 1.0e-70
            stopphi.terminal = True
            stopphi.direction = -1

            if self.MPBHi >= 8.:
                if self.bPBHi > -15.:
                    atol=1.e-5
                    meth='BDF'
                else:
                    atol=1.e-2
                    meth='Radau'
            else:
                atol=1.e-15
                meth='BDF'
            
            # solve ODE
            #solFBE = solve_ivp(lambda t, z: FBEqs(t, z, nphi, mDM, sDM, Mi, xilog10),
            #                   [np.log10(Amaxin), xflog10], v0, method=meth, \
            #                   events=[StopM,stopphi], rtol=1.e-7, atol=atol, max_step=0.1) 

            solFBE = solve_ivp(lambda t, z: FBEqs(t, z, nphi, mDM, sDM, Mi, xilog10),
                               [np.log10(Amaxin), xflog10], v0, method=meth, \
                               events=StopM, rtol=1.e-7, atol=atol, max_step=0.1) 

            if solFBE.t[-1] < 0.:
                print(solFBE)
                print(afw, tau, 1.05*xflog10)
                break

            # Concatenating solutions
            
            xBE    = np.append(xBE,    solFBE.t[:] + xilog10)
            MBHBE  = np.append(MBHBE,  solFBE.y[0,:])
            astBE  = np.append(astBE,  solFBE.y[1,:])
            PhikBE  = np.append(PhikBE,  solFBE.y[2,:])
            RadBE  = np.append(RadBE,  solFBE.y[3,:])
            PBHBE  = np.append(PBHBE,  solFBE.y[4,:])
            TBE    = np.append(TBE,    solFBE.y[5,:])
            NDMHBE = np.append(NDMHBE, solFBE.y[6,:])
            tmBE   = np.append(tmBE,   solFBE.y[7,:])

            # Updating values of initial parameters
            
            Mi    = solFBE.y[0,-1]
            asi   = solFBE.y[1,-1]
            rhophi_in = solFBE.y[2,-1]
            rRadi = solFBE.y[3,-1]
            rPBHi = solFBE.y[4,-1]
            Ti    = solFBE.y[5,-1]
            NDMHi = solFBE.y[6,-1]
            ti    = solFBE.y[7,-1]
            
            xilog10 += solFBE.t[-1]

            i += 1

            if i > 100:
                xflog10 = xilog10
                print("I'm stuck!", Mi, bi)
                print()
                break
        else:
            xflog10 = xilog10    # We update the value of log(a) at which PBHs evaporate


        Tev = TBE[-1]
        
        
        print('------------------')
        print('    log(a) after evaporation = ', solFBE.t[-1], '    ')
        print('------------------')


        #-----------------------------------------#
        #           After BH evaporation          #
        #-----------------------------------------#
        def stopphi2(t, y):
            return y[1]*y[0]**(-(6*k/(k+2))) - 1.0e-70
        stopphi2.terminal = True
        stopphi2.direction = -1
        
        Tfin = 1.e-3 # Final plasma temp in GeV
        
        xzmax = xflog10 + np.log10(np.cbrt(bh.gstarS(TBE[-1])/bh.gstarS(Tfin))*(TBE[-1]/Tfin))
        xfmax = max(xflog10, xzmax)

        v0aBE = [tmBE[-1], PhikBE[-1], RadBE[-1], TBE[-1], NDMHBE[-1]]
        
        # solve ODE        
        #solFBE_aBE = solve_ivp(lambda t, z: FBEqs_aBE(t, z), [xflog10, xfmax], v0aBE, \
        #                       events=stopphi2, method='BDF', rtol=1.e-5, atol=1.e-10, max_step=0.1)

        solFBE_aBE = solve_ivp(lambda t, z: FBEqs_aBE(t, z), [xflog10, xfmax], v0aBE, \
                               method='BDF', rtol=1.e-5, atol=1.e-10, max_step=0.1)

        npaf = solFBE_aBE.t.shape[0]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #       Joining the solutions before and after evaporation       #
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        x    = np.concatenate((xBE, solFBE_aBE.t[:]), axis=None)
 
        MBH  = np.concatenate((MBHBE,  np.zeros(npaf)), axis=None)
        ast  = np.concatenate((astBE,  np.zeros(npaf)), axis=None)
        t    = np.concatenate((tmBE,   solFBE_aBE.y[0,:]), axis=None) 
        Phik = np.concatenate((PhikBE,  solFBE_aBE.y[1,:]), axis=None)  
        Rad  = np.concatenate((RadBE,  solFBE_aBE.y[2,:]), axis=None)    
        PBH  = np.concatenate((PBHBE,  np.zeros(npaf)),  axis=None)
        T    = np.concatenate((TBE,    solFBE_aBE.y[3,:]), axis=None)
        NDMH = np.concatenate((NDMHBE, solFBE_aBE.y[4,:]), axis=None)
                
        return [x, t, MBH, ast, Phik, Rad, PBH, T, NDMH, Tev]

    #------------------------------------------------------------#
    #                                                            #
    #                     Conversion to Oh^2                     #
    #                                                            #
    #------------------------------------------------------------#
    
    def Omega_h2(self):
        '''
        This function directly returns Omega_h2, using the solution above
        '''

        x, t, MBH, ast, Phik, Rad, PBH, TUn, NDMH, Tev = self.Solt()
        
        nphi = (2.*zeta(3)/np.pi**2)*TUn[0]**3             # Initial photon number density
        
        rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18        # Critical density in GeV^3
        
        T0 = 2.34865e-13                                   # Temperature today in GeV
        
        Oh2  = NDMH[-1] * nphi * 10.**(-3.*x[-1]) * 10.**self.mDM * (bh.gstarS(T0)/bh.gstarS(TUn[-1]))*(T0/TUn[-1])**3*(1/rc)

        return Oh2
        
        

        
