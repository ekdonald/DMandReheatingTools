#!/bin/bash

# For the moment k=2 is not supported since PBH would form in matter dominated universe
kList=("4")  # list of powers of inflaton potential to scan over


for kvar in "${kList[@]}"; do
	kn=$kvar
	
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

	path="./Results/k="$kvar
	yuklist=("1E00")        # yukawa coupling of  Φ --> f f  (y_eff Φff)
	yukdir=("yukawa=1E00")  # directory for storing data for each yukawa coupling

	logMBHinlist=("2" "6")  # log10 of initial black hole masses to consider  (in gramms, g)
	Mdist="ext"             # mass distribution ("ext": enxtended power-law, "mono": monochromatic)
	sigmaM="2"              # sigma = 0 for monochromatic BH "mono"
	betalist=("-10" "-14")  # Initial fractions of black hole energy density to scan over
	
	# log10 of dark matter particle mass (in GeV)
	#logmDM="5"
	logmDMList=("4" "6")  
		
	kminus=$((kn-2))
	kplus=$((kn+2))
	omega=$(echo "scale=5; ($kminus / $kplus)" | bc -l)        
	alpha=$(echo "scale=5; (((4*$omega+2))/(($omega+1)))" | bc -l)  

	echo "kn, path, omega, alpha =" $kn, $path, $omega, $alpha
	ip=$((0))

	for yuk in "${yukdir[@]}"; do
		rm tempyuk.dat
		yphi="${yuklist[$ip]}"
		echo "ip, yphi, yuk = ", $ip, $yphi, $yuk
		
		for logMBHin in "${logMBHinlist[@]}"; do
			MBHin="MBH1E"$logMBHin
			touch tempyuk.dat
			echo $yphi	$kn	$logMBHin > tempyuk.dat
			
			for logmDM in "${logmDMList[@]}"; do
				mDMin="mDM1E"$logmDM			
			
				for beta in "${betalist[@]}"; do
					echo "k, yuk, log10_mDM, log10_Min, beta = " $kn  $yphi  $logmDM $logMBHin $beta
					mkdir -p $path/phiff/$yuk/$MBHin/$mDMin/databeta=$beta/sigma_$sigmaM/
					python3 -W ignore example_DM_MassDist.py $logmDM $beta $logMBHin $sigmaM $alpha $Mdist $yphi
					mv $path/data_scan*  $path/phiff/$yuk/$MBHin/$mDMin/databeta=$beta/sigma_$sigmaM/
				done
			done
		done
		ip=$((ip+1))
	done
done


