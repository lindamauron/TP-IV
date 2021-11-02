Explaination of different files

Classes : 
- ExactIsing1D.py (exact solution of the 1D Ising model)
- MeanField.py (Mean field approximation H = - sum bk sk)
- Jastrow.py (Jastrow model H = sum si Wij sj)

- MCMC (engine to run MCMC loops)


Tests of classes : 
- Test_IsingModel.py
- Test_MeanFieldModel.py
- Test_Jastrow.py
- Test_MonteCarloMarkovChains.py

Analysis : 
- VMC_<class>.py (VMC using MeanField or Jastrow model and MCMC)

Results :
- *.png : graphs of different simulations (all info in name + title)

Folders : 
- Old code : scripts before the clean-up implementation + graphs of exact Ising1D)
