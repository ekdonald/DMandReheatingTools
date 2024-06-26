
# ABOUT THE PACKAGE

Python package providing the solution of the Friedmann - Boltzmann equations for Primordial Black 
Holes + SM radiation + BSM Models + inflaton field.

FRISBHEE - FRIedmann Solver for Black Hole Evaporation in the Early-universe
Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner

# INCLUDING INFLATON

Author: D. Kpatcha 


# FRISBHEE + Inflaton + Reheating + Dark Matter

Modificatins to include inflaton in the evolution
The main part is FRISBHEE has been modified to include the decay/scattering of inflaton in the dynamical
evolving system. It allows to study all scenarios of reheating, and compute the relic density of dark matter
produced by the evaporating PBH. Inflaton decays into SM only.

This package provides the solution of the Friedmann - Boltzmann equations for Primordial Black 
Holes + SM radiation + BSM Models. We consider the collapse of density fluctuations as the PBH 
formation mechanism. We provide codes for monochromatic and extended mass and spin distributions.


# CONTAINS

	SolFBEqs_MassDist.py: contains classes related to solving equations for extended PBH mass 
	                      distribution return the full evolution of the PBH, SM and Dark Radiation 
                              comoving energy densities, together with the evolution of the PBH mass 
                              and spin as function of the scale factor. 
                              
                               #  Distribution types: (see "example_DM_MassDist.py")
				#    0 -> Lognormal, requires 1 parameter, sigma
				#    1 -> Power law, requires 2 parameters, [sigma, alpha]
				#    2 -> Critical collapse, doesn't require any parameters
				#    3 -> Metric preheating, doesn't require any parameters
			
				
	SolFBEqs_Mono.py: same as "SolFBEqs_MassDist.py" but for monochromatic mass distribution
                      
	Integrator.py: mathematical functions

	Functions_phik.py: Contains essential functions related to the evolution of the inflaton.

	BHProp.py: contains essential constants and parameters

	example_DM_MassDist.py: main programm of the package. Read parameters from the command line input
                                can be run in terminal (see "run.sh" or "run_script.sh")

	scan_beta.py: An example of python script for scan over different values of parameters leading to correct 
	              relic density

	run.sh: An example of bash shell script to run the programm for specific PBH initial mass and infaton potential.
                input paramters (e.g initial fraction of BH energy density beta, infaton decay coupling, ...) 
                are defined.

	run_script.sh: An example of bash shell script to run the programm for various (scan) PBH initial mass and
               	infaton potential. input paramters (e.g initial fraction of BH energy density beta, 
               	infaton decay coupling, ...) are defined.


# INSTALLATION AND RUNNING

	1. No specific installation instructions needed. just required modules (see below)
	
	2. For running see "./run.sh" or "./run_script.sh"
	   The results (output files) are stored in 
	   ./Results/k=$kn/phiff/$yuk/$MBHin/databeta=$beta/sigma_$sigmaM/


# REQUIRED MODULES

	1. We use Pathos (https://pathos.readthedocs.io/en/latest/pathos.html) for parallelisation, and 
           tqdm (https://pypi.org/project/tqdm/) for progress meter. These should be installed in order 
           to FRISBHEE-Inflaton to run.
        
	2. ulysses (provided with the package, but can be installed independtly)


# CREDITS

If using this code, please cite:

    1. arXiv:2107.00013, arXiv:2107.00016, arXiv:2207.09462, arXiv:2212.03878
    
    2. Including inflaton: arXiv:2305.10518, arXiv:2309.06505

