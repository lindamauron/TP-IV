T=1_singlepeak

T = 1
mu1=-2
mu2=5 
sigma1=1
sigma2=2
prop = 0
lr = 0.05

nsteps = int(5e3)
Ns = 5000

sampling of 5000 samples : mean 5.16, variance 1.23
############################################################################
T=1_twopeaks

T = 1
mu1=-3
mu2=5 
sigma1=1
sigma2=2
prop = 0.5
lr = 0.05

nsteps = int(1e3)
Ns = 5000


sampling of 5000 samples : mean -3.16, variance 1.37

##############################################################################
T=1_twopeakssame

T = 1
mu1=-3
mu2=5 
sigma1=1
sigma2=1
lr = 0.05

nsteps = int(1e3)
Ns = 5000

mean ~-3, variance ~1.2

###################################################################################
T = 4 or 2
mu1=-3
mu2=5 
sigma1=1
sigma2=1
prop = 0.5
lr = 0.01

nsteps = int(100)
Ns = 10000
#####################################################################################
T = 5
mu1=-3
mu2=5 
sigma1=1
sigma2=1
prop = 0.5
lr = 0.01

nsteps = int(100)
Ns = 10000


p_to_approach = Distribution(mu1,mu2,sigma1,sigma2,prop=prop)


features = [[5,7,3,1], [5,7,3,1], [5,7,3,1], [3,5,7,3,1], [5,7,3,1]]

mean -2.84, variance 1.42
