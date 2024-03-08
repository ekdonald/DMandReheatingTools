import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib as mpl
import scipy as sp
import matplotlib.colors as colors
from matplotlib import rc
import math
import os
import subprocess
from matplotlib.colors import Normalize
from matplotlib import rcParams
rcParams['text.usetex'] = True
from sympy.solvers import solve
from sympy import Symbol, expand
from scipy.interpolate import griddata
import random
from scipy.special import gamma, factorial
from sympy import ImageSet, S , solveset
import math as math 
import sympy as sym
from sympy import *
import importlib
import sys
import warnings
warnings.filterwarnings('ignore')
warnings.warn("", FutureWarning)

import BHProp as bh #Schwarzschild and Kerr BHs library



def readyuk():
	'''
	 read yukawa (phi to f f coupling), k value in phi^k inflation, and MBHin 
	 yphi, kn, logmDM, MBHin
	'''
	ff = pd.read_csv("tempyuk.dat",names=["yuk", "kvar", "logmDM", "Min"], delim_whitespace=True,header=None)
	return ff["yuk"][0], ff["kvar"][0], ff["logmDM"][0], ff["Min"][0]


class phik_process:
    '''
    Define all the functions needed to compute the parameters in phi^k inflation 
    '''
    def __init__(self):
        self.process = "phiff"      # process = "phiff", "phibb", "phiphibb"
        self.Mp_real = 1.22e19 
        self.Mp = 2.435e18 
        self.geff = 100.  # 427./4.
        self.alpha_0 = 1 
        self.alpha_1 = np.sqrt(2/(3*self.alpha_0)) 
        self.AR_CMB = 2.19e-09 
        self.ns_CMB = 0.9645
        self.alphaStar = self.geff*np.pi**2/30.
        self.yeff = readyuk()[0] 
        self.mueff = 1.0e06
        self.sigmaeff = 1.0e-6
        self.kvar = readyuk()[1]    
        self.Min  = readyuk()[3] 


    def lparameter(self, k):
        '''
        l funtion in the decay rates of the inflaton
        '''        
        if self.process == "phiff": res = 1/2-1/k
        elif self.process == "phibb": res = 1/k-1/2
        elif self.process == "phiphibb": res = 3/k-1/2
        else:
            print("Invalid process to find l. Using the default process phi-->ff")
            res = 1/2-1/k
        return res


class phik_funcs:
    def __init__(self, process):
        '''
         Inflaton decay process: "phiff", "phibb", or "phiphibb"
        '''
        self.process = phik_process().process


    def gammaphi(self, k, lambdac):
        '''
        gamma funtion for the decay of the inflaton
        '''        
        if self.process == "phiff": 
            res = np.sqrt(k*(k-1))*np.power(lambdac,1/k)*phik_process().Mp*phik_process().yeff**2/(8*np.pi)
        elif self.process == "phibb": 
            res = phik_process().mueff**2/(8*np.pi*np.sqrt(k*(k-1))*np.power(lambdac,1/k)*phik_process().Mp)
        elif self.process == "phiphibb": 
            res = phik_process().sigmaeff**2*phik_process().Mp/(8*np.pi*np.power(k*(k-1),3/2)*np.power(lambdac,3/k))
        else:
            print("Invalid process to find gamma. Using the default process phi-->ff")
            res = np.sqrt(k*(k-1))*np.power(lambdac, 1/k)*phik_process().Mp*phik_process().yeff**2/(8*np.pi)
        return res


    def lparameter(self, k):
        '''
        l funtion in the decay rates of the inflaton
        '''        
        if self.process == "phiff": res = 1/2-1/k
        elif self.process == "phibb": res = 1/k-1/2
        elif self.process == "phiphibb": res = 3/k-1/2
        else:
            print("Invalid process to find l. Using the default process phi-->ff")
            res = 1/2-1/k
        return res;


    def Gkl(self, k, lambdac):
        '''
        Gkl functions
        '''            
        lparm = phik_funcs(self.process).lparameter(k)
        gammaparam = phik_funcs(self.process).gammaphi(k,lambdac)
        factor = np.sqrt(3.0)*gammaparam*2*k/(k+8-6*k*lparm)
        rhoend = phik_funcs(self.process).rhoendvar(k)
        p1 = rhoend**(lparm+1/2)
        p2 = phik_process().Mp**(4*lparm-1)
        res = factor*p1/p2
        return  res;
        
 
    def rCMBvar(self, k):
        '''
        r_parameter with CMB constraints
        '''          
        fact = 48*phik_process().alpha_0*k**2*(1-phik_process().ns_CMB)**2
        p1 = (2*k+np.sqrt(4*k**2+6*phik_process().alpha_0*k*(2+k)*(1-phik_process().ns_CMB)))**2
        res = fact/p1
        return res;
    
    
    def lambdavar(self, k):
        '''
        lambda parameter in V = lambda*phi^k/phik_process().Mp^(4-k)
        '''              
        r_CMB = phik_funcs(self.process).rCMBvar(k)
        fact = np.power(phik_process().alpha_1, k)*3*np.pi**2*phik_process().AR_CMB*r_CMB/2
        p1 = (k*(1+k)+np.sqrt(k**2+3*phik_process().alpha_0*(2+k)*(1-phik_process().ns_CMB)))/(k*(2+k))
        p2 = np.power(p1, k)
        res = fact*p2
        return res;


    def rhoendvar(self, k):
        '''
        Energy density at the end of inflation 
        '''              
        lambdac = phik_funcs(self.process).lambdavar(k)
        res = 3*lambdac*(phik_process().Mp/phik_process().alpha_1)**4*(np.power(k/(k+np.sqrt(3*phik_process().alpha_0)),k))/2
        return res;
    
    
	
    def Hvar(self, rhophi, rhoRad, rhoPBH):
        '''
        Hubble parameter as function of rhophi + rhoRad 
        '''   
        res = np.power(rhophi + rhoRad + rhoPBH, 1/2)/(np.sqrt(3)*phik_process().Mp)
        return res    
    
	
    def rhophi(self, k, A, rhoend):
        '''
        Energy density of the inflaton
        '''           
        res = rhoend*np.power(A,-6*k/(k+2))
        return res
    

    def rhoRad(self, k, A):
        '''
        Energy density of the radiation
        '''           
        lparm = phik_funcs(self.process).lparameter(k)
        lambdac = phik_funcs(self.process).lambdavar(k)
        Gklvar  = phik_funcs(self.process).Gkl(k, lambdac)
        res = Gklvar*np.power(A,-4.)*(np.power(A,-(6*k*lparm-k-8)/(k+2))-1)
        return res
   

    def omegaphi(self, k):
        '''
        Equation of state of radiation
        '''       
        return (k-2)/(k+2)


    def rhoRmax(self, l, rhoend, Amax, Hend, gammaparam):
        '''
        Radiation energy density when T = T_max
        '''   
        fact = 2/(3+6*l)
        p1 = gammaparam/Hend
        p2 = np.power(rhoend,l+1)/np.power(phik_process().Mp,4*l)
        res = fact*p1*p2*np.power(Amax,-4)
        return res

    '''
    def m_pbh_in(self, k):
        lambdac = phik_funcs(self.process).lambdavar(k)
        rhoend = phik_funcs(self.process).rhoendvar(k)
        omega = (k-2)/(k+2)
        rhophi = rhoend*np.power(self.Amaxin,-6*k/(k+2))
        rhoradi = phik_funcs(self.process).rhoRad(k, self.Amaxin)
        eff = np.power(omega,3/2)
        H = np.power(rhophi+rhoradi,1/2)/(np.sqrt(3)*phik_process().Mp)
        res = 4*np.pi*eff*phik_process().Mp**2/H
        return res;
    '''


    def m_pbh_solvein(self, k, Ain):
        lambdac = phik_funcs(self.process).lambdavar(k)
        rhoend = phik_funcs(self.process).rhoendvar(k)
        omega = (k-2)/(k+2)
        rhophi = rhoend*np.power(Ain,-6*k/(k+2))
        
        lparm = phik_funcs(self.process).lparameter(k)
        Gklvar  = phik_funcs(self.process).Gkl(k, lambdac)
        if (k < 7):
            rhoradi = Gklvar*np.power(Ain,-4.)*np.power(Ain,-(6*k*lparm-k-8)/(k+2))
        else:
            rhoradi = Gklvar*np.power(Ain,-4.)*(-1)            
        
        eff = np.power(omega,3/2)
        H1 = np.power(rhophi,1/2)/(np.sqrt(3)*phik_process().Mp)
        resphi = 4*np.pi*eff*phik_process().Mp**2/H1
        
        H2 = np.power(rhoradi,1/2)/(np.sqrt(3)*phik_process().Mp)
        resrad = 4*np.pi*eff*phik_process().Mp**2/H2
        
        return resphi, resrad;

 
    #-------------------------------     
    def Amax(self):  
        Ain = Symbol("Ain", positive=True)
        ksol = phik_process().kvar
        resphi = solve(phik_funcs(self.process).m_pbh_solvein(ksol, Ain)[0]*bh.GeV_in_g - 10**(phik_process().Min), Ain)
        resrad = solve(phik_funcs(self.process).m_pbh_solvein(ksol, Ain)[1]*bh.GeV_in_g - 10**(phik_process().Min), Ain)
        
        print("resphi, resrad", resphi, resrad)

        if (len(resphi) == 0):
            print("No solution of Ain. Using default solution.")
            Ainphi = 1.
        else:
            Ainphi = resphi[-1]
            
        print("Ainphi = ", Ainphi)
        radin = phik_funcs(phik_process().process).rhoRad(ksol, Ainphi)
        rhoendv = phik_funcs(phik_process().process).rhoendvar(ksol)
        phiin = phik_funcs(phik_process().process).rhophi(ksol, Ainphi, rhoendv) 
        
        if(radin < phiin):
            res = Ainphi
        else:
        
            if (len(resrad) == 0):
                print("No solution of Ain. Using default solution.")
                res = Ainphi
            else:
                res = resrad[-1]

        print("Ainphi, Ainrad, Ain = ", Ainphi, resrad[-1], res)
        radin = phik_funcs(phik_process().process).rhoRad(ksol, res)
        rhoendv = phik_funcs(phik_process().process).rhoendvar(ksol)
        phiin = phik_funcs(phik_process().process).rhophi(ksol, res, rhoendv)
        
        print("phiin, radin", phiin, radin)        
        
        return res;
                

class Amax_funcs:
    def __init__(self):
        '''
        Energy density at the end of inflation 
        '''           
        self.Amaxin = float(phik_funcs(phik_process().process).Amax())
    

def readoutputfiles():
    kk = phik_process().kvar
    pathb = "./Results/k={}".format(kk)

    incols = ["logbeta","logMc","sigma_M","alpha","rhprocess","kvar","yeff","mueff",\
              "sigmaeff", "tag", "logmDM", "Tev"]
    data_inParams = pd.read_csv(pathb+"/data_scan_inparameters.dat",sep=" ", names=incols)

    #--------- Read data
    beta = 10.**data_inParams["logbeta"].iloc[0]
    Mc = 10.**data_inParams["logMc"].iloc[0]
    sigma_M = data_inParams["sigma_M"].iloc[0]
    alpha = -data_inParams["alpha"].iloc[0]
    tag = data_inParams["tag"].iloc[0]
    yeff = data_inParams["yeff"].iloc[0]
    mueff = data_inParams["mueff"].iloc[0]
    sigmaeff = data_inParams["sigmaeff"].iloc[0]
    #kk = data_inParams["kvar"].iloc[0]
    rhprocess = data_inParams["rhprocess"].iloc[0]

    pre_name = pathb+'/data_scan_'
    tag_data = '_logbeta_'+str(int(np.log10(beta)))+'_logMc_'+str(int(np.log10(Mc)))\
               + '_sigM_' +str(int(sigma_M))+ '_siga_' + '_alpha_' +str(int(alpha))

    data_A = pd.read_csv(pre_name+str(tag)+'_log10a'+tag_data+'.txt',header=None, names=["Log10_A"])
    data_rRad = pd.read_csv(pre_name+str(tag)+'_a4rhorad'+tag_data+'.txt',header=None, names=["a4rhorad"])
    data_rPBH = pd.read_csv(pre_name+str(tag)+'_a3rhoPBH'+tag_data+'.txt',header=None, names=["a3rhoPBH"])
    data_TPBH = pd.read_csv(pre_name+str(tag)+'_TPBH'+tag_data+'.txt',header=None, names=["TPBH"])
    data_rPhi = pd.read_csv(pre_name+str(tag)+'_funkrhophi'+tag_data+'.txt', header=None, names=["afunPhi"])
    data_a3nDM = pd.read_csv(pre_name+str(tag)+'_a3nDM'+tag_data+'.txt',header=None, names=["a3nDM"])
    data = pd.concat([data_A, data_rPhi, data_rRad, data_rPBH, data_TPBH,data_a3nDM], axis=1).astype("float")

    return [data,  data_inParams];


def findARH():
    kk = phik_process().kvar
    data = readoutputfiles()[0]
    pos1List = np.where(data.a3rhoPBH/10**(3*data.Log10_A) \
                         + data.afunPhi/np.power(10**(data.Log10_A),6*kk/(kk+2))
                         - data.a4rhorad/10**(4*data.Log10_A) >= 0)
    if len(pos1List[0])==0:
        pos = -1
        ARH = 10**(data.Log10_A[pos])
        print("PBH evaporates before BBN")
    else:
        pos = pos1List[0][-1] 
        ARH = 10**(data.Log10_A[pos])
    return ARH;


def findTRH():
    kk = phik_process().kvar
    data = readoutputfiles()[0]
    pos1List = np.where(data.a3rhoPBH/10**(3*data.Log10_A)+ data.afunPhi/np.power(10**(data.Log10_A),6*kk/(kk+2))
                        > data.a4rhorad/10**(4*data.Log10_A))
    if len(pos1List[0])==0:
        pos = -1 
        rhoRadRH = data.a4rhorad.iloc[pos]/10**(4*data.Log10_A.iloc[pos])
        TRH = np.power(rhoRadRH/phik_process().alphaStar, 1/4)  
        print(" In RD ..., pos =", pos)
    else:
        pos = pos1List[0][-1] 
        rhoRadRH = data.a4rhorad[pos]/10**(4*data.Log10_A[pos])
        TRH = np.power(rhoRadRH/phik_process().alphaStar, 1/4)  
    return TRH;
