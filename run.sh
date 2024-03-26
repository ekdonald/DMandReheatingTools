#!/bin/bash

# For the moment k=2 is not supported since PBH would form in matter dominated universe
kn='4'                 # power of inflaton potential
path="./Results/k="$kn
mkdir -p $path          # directory for storing data for each yukawa coupling
yphi="1E00"            # yukawa coupling of  Φ --> f f  (y_eff Φff)
yukdir="yukawa=1E00"   

logMBHin="2"  # log10 of initial black hole mass  (in gramms, g)
sigmaM="2"    # sigma = 0 for monochromatic BH "mono"
Mdist="ext"   # mass distribution ("ext": enxtended power-law, "mono": monochromatic)
beta="-10"    # Initial fraction of black hole energy density
tag="0"       # a tag 

logmDM="5"    # log10 of dark matter particle mass (in GeV)
	
kminus=$((kn-2))
kplus=$((kn+2))
omega=$(echo "scale=5; ($kminus / $kplus)" | bc -l)        
alpha=$(echo "scale=5; (((4*$omega+2))/(($omega+1)))" | bc -l)  

echo "#-------------------------------------------------------#"
echo "#       PBH  +  Φ  with potential V(Φ) = Φ^"$kn"            #"
echo "#                                                       #"
echo "# In this version k=2 is not supported, since PBH would #"
echo "#         form in matter dominated universe.            #"
echo "#                    Chose k > 2                        #"
echo "# PBH evaporate to SM and Dark matter. Compute the      #"
echo "#           relic density of dark matter                #"
echo "#-------------------------------------------------------#"
echo " "

echo "omega, yphi, alpha = ", $omega, $yphi, $alpha

touch tempyuk.dat
echo $yphi    $kn    $logmDM    $logMBHin > tempyuk.dat
			
mkdir -p $path/phiff/$yuk/$MBHin/databeta=$beta/sigma_$sigmaM/
python3 -W ignore example_DM_MassDist.py $logmDM $beta $logMBHin $sigmaM $alpha $Mdist $yphi
mv $path/data_scan*  $path/phiff/$yukdir/$MBHin/databeta=$beta/sigma_$sigmaM/
