Explaination of different files

Classes : 
- Models:- MeanField.py (Mean field approximation psi(s) = exp(sum bi si)
	 - Jastrow.py (Jastrow model psi(s) = exp (sum si Wij sj) )
- DiscreteOperator : contains the hamiltonians
	- IsingTransverse H = -J sum Z_i Z_i+1 -h sum X_i
	- X,Y,Z : Pauli matrices

- MCMC (engine to run MCMC loops)


Tests of classes : 
- Test_Models : testing the implementation of the models


Analysis : 
- VMC_<class>.py (VMC using MeanField or Jastrow model and MCMC)

Results :
- *.png : graphs of different simulations (all info in name + title)

Folders : 
- Ising_Analytical : graphs on the exact analytical results of the Ising model in 1D
- ClassicalCode : Code and results of the VMC on MeanField and Jastrow model
		  The Jastrow partition function is only correct for N_spins <= 3
- Angles_canceled : project about continuous space of angles : won't continue
- VMC : Codes for quantum MCMC (Semester 1)