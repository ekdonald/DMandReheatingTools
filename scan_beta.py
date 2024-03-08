#!/usr/bin/env python3
#import Functions_Kerr_Power_Law as Fun
import multiprocessing
import multiprocess as mp
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import BHProp as bh #Schwarzschild and Kerr BHs library
import Functions_phik as phik_functions
import sys
import os 


# above betacritPBHdomination, it is exclusively PBH rehearting
def betacritBH(Min, n):
    Mplanck = 2.435e18 
    GeV_to_g  = 1.782661907e-24  # 1 GeV in g
    geff = 427/4
    epsilon = np.pi*geff/480
    w = (n-2)/(n+2)
    gammav = np.power(w, 3/2)
    p0 = np.power(epsilon/(2*np.pi*gammav*(1+w)), 2*w/(1+w))
    p1 = np.power(Mplanck*GeV_to_g/Min, 4*w/(1+w))
    res = p0*p1
    return res;



kList = [4]


for kvar in kList:
    print("Scanning for critical beta for k = {}".format(kvar))
    kn = kvar
    path="./Results/k={}".format(kvar)

    #os.system('mv %s  %s' % (path, path+"_nono") )
    os.system('mkdir -p %s' % (path))
    
    yukdir  = ["yukawa=1E-10"]
    yuklist = [1.0E-10] 
    betalist = [-10 -12, -14, -16]

    BRDM    = 1.0E-02  # not being used
    logMBHinList = [6, 2]

    Mdist = "ext"
    sigmaM = 2
    kminus = kn-2
    kplus = kn+2
    omega = kminus/kplus      
    alpha= (4*omega+2)/(omega+1)

    print("kn, path, omega, alpha =", kn, path, omega, alpha)
    ip = 0

    # Over Yukawa
    for yuk in yukdir: 
        os.system('rm tempyuk.dat')
        yphi = yuklist[ip]
        print("ip, yphi, yuk = ", ip, yphi, yuk)

        # Over MBHin
        for logMBHin in logMBHinList:
            print("MBHin = ", 10**logMBHin)
            mBHdir = "MBH1E{}".format(logMBHin)
            MBHin = 10**logMBHin
            
            os.system('touch tempyuk.dat')
            print("logMBHin, mBHdir = ", logMBHin, mBHdir)

            # Over beta
            for beta in betalist:
                Oh2 = 0.0
                logmDMmax = 20
                logmDMmin = 0
                
                # Check Oh^2
                while ((Oh2 < 0.11) | (Oh2 > 0.13)) & (logmDMmax-logmDMmin > 1.0e-02):
                    os.system('rm -f %s/data_scan*' %(path))

                    logmDM = (logmDMmax + logmDMmin)/2
                    os.system('echo %s %s %s %s > tempyuk.dat' % (yphi, kn, logmDM, logMBHin))
                    mDMindir = "mDM1E{}".format(str(logmDM))
                    print("beta, logMBHin, logmDM = ", beta, logMBHin, logmDM)

                    os.system('python3 -W ignore example_DM_MassDist.py %s %s %s %s %s %s %s' \
                              % (logmDM, beta, logMBHin, sigmaM, alpha, Mdist, yphi))
		        
                    # COMPUTE Omegah2
                    ff = pd.read_csv("Omegah2.dat",names=["Oh2"], delim_whitespace=True,\
                                      header=None).astype("float")
                    Oh2 = ff["Oh2"].iloc[0]

                    print("#-----------------------------------------------#")
                    print(" Oh2 ", Oh2)
                    print("#-----------------------------------------------#")

                    if (Oh2 < 0.11):
                        logmDMmax = logmDM   # to be checked
                    elif (Oh2 > 0.13):
                        logmDMmin = logmDM    # to be checked

                # Store data  
                os.system('mkdir -p  %s/%s/%s/%s/beta=%s/%s/databeta=%s/sigma_%s/' % (path, "phiff", yuk, mBHdir, beta, mDMindir, beta, sigmaM))
                os.system('mv %s/data_scan*  %s/%s/%s/%s/beta=%s/%s/databeta=%s/sigma_%s/' % (path, path, "phiff", yuk, mBHdir, beta, mDMindir, beta, sigmaM))
        #
        ip = ip+1

