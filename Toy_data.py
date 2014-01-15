# This program uses a number of delay estimation methods to benchmark the 
# convolved Gaussian process framework for RNA pol-II dynamics
#
# The delay estimation techniques used are
#	1) Cross correlation (Corr)
#	2) The discrete correlation function method (DCF) 
#	3) A Kernel method (Kern)
#	4) A GP method without convolution (GP-NoConv)
#	5) The proposed GP method with convolution (GP-Conv)
#
# Ciira wa Maina, 2014
# Dedan Kimathi University of Technology.
# Nyeri-Kenya

import sys
import numpy as np
import scipy as sp
import pylab as pb
from scipy.optimize import fmin_tnc
import delay_gp_funcs as dgf#GP delay estimation functions without convolution 
import conv_gp_funcs as cgf #Convolved GP functions
import time
from scipy.stats.mstats import mquantiles


def dcf(X,Y,T,num_bin,noise_std):

	'''
	This function implements the discrete correlation function 
	(DCF) delay estimation method described in Edelson RA, Krolik JH (1988) "The discrete correlation function - A new method for 		analyzing unevenly sampled variability data." The Astrophysical Journal 333: 646-659.

	'''

	#obtain the delta Ts
	deltaT=T[:,None]-T[None,:]
	#iu1 = np.triu_indices(len(T),1)
	iu1 = np.triu_indices(len(T))
	hist, bin_edges=pb.histogram(np.abs(deltaT[iu1]),num_bin)
	cent=bin_edges[0:len(bin_edges)-1]+np.diff(bin_edges)*.5
	dcf=np.zeros(len(cent))
	sigx=np.var(X)
	sigy=np.var(Y)
	muX=np.mean(X)
	muY=np.mean(Y)
	for i in range(0,len(cent)):
		for j in range(0,len(T)):
			for k in range(j,len(T)):
				if i<len(cent)-1:
					if (np.abs(deltaT[j,k])>=bin_edges[i])&(np.abs(deltaT[j,k])<bin_edges[i+1]):
						dcf[i]+=((X[j]-muX)*(Y[k]-muY))/np.sqrt((sigx-noise_std**2)*(sigy-noise_std**2))
				elif i==len(cent)-1:
					if (np.abs(deltaT[j,k])>=bin_edges[i])&(np.abs(deltaT[j,k])<=bin_edges[i+1]):
						dcf[i]+=((X[j]-muX)*(Y[k]-muY))/np.sqrt((sigx-noise_std**2)*(sigy-noise_std**2))


	dcf[hist>0]=dcf[hist>0]/hist[hist>0]
	return cent[np.argmax(dcf)],dcf,cent


def kernDelay(Data,t_obs,num_pt,num_seg,noise_std):


	'''
	This function computes the delay estimate using the kernel method described in 
	Cuevas-Tello JC, Tino P, Raychaudhury S (2006) "How accurate are the time delay estimates in gravitational
	lensing?" Astronomy and Astrophysics 454: 695-706.
	'''


	T=np.linspace(t_obs[0],t_obs[-1],num_pt)
	w_max=t_obs[-1]
	w_min=t_obs[-1]*.0625
	W=np.linspace(w_min,w_max,num_pt)
	num_obs=len(t_obs)

	#LOO cross validation
	BestW=np.zeros(num_pt)
	cen_cv=t_obs
	num_mix_cv=len(t_obs)
	for w in range(num_pt):
		width_est=W[w]*np.ones(num_mix_cv)
		R=np.zeros(num_obs)
		for ii in range(num_obs):
			indx=np.nonzero(np.arange(num_obs)!=ii)
			Q=np.zeros(num_pt)

			Y_cv=[]
			for i in range(0,num_seg):
	
				Y_cv=np.concatenate((Y_cv,(Data[i,indx])[0]))

			Y_cv=Y_cv/noise_std

			for i in range(num_pt):
				K=compute_K(t_obs[indx],cen_cv,width_est,T[i],num_seg)#use k-1 obs
				K=K/noise_std
				alpha_est=np.dot(np.linalg.pinv(K),Y_cv[:,None])
				sig_est=np.zeros((num_seg,len(t_obs)))
				for j in range(num_mix_cv):

					for k in range(num_seg):
						sig_est[k,:]+=alpha_est[j]*computeKern(t_obs-cen_cv[j]-k*T[i],width_est[j])
	

				Q[i]=np.sum((Data[:,ii]-sig_est[:,ii])**2/noise_std**2)

			R[ii]=np.min(Q)
		BestW[w]=np.mean(R)

		
	#Use LOOCV result
	Y=[]
	for i in range(0,num_seg):

		Y=np.concatenate((Y,Data[i,:]))

	Q=np.zeros(num_pt)
	width_est=W[np.argmin(BestW)]*np.ones(num_mix_cv)
	for i in range(num_pt):
		K=compute_K(t_obs,cen_cv,width_est,T[i],num_seg)
		K=K/noise_std
		alpha_est=np.dot(np.linalg.pinv(K),Y[:,None])
		sig_est=np.zeros((num_seg,len(t_obs)))
		for j in range(num_mix_cv):

			for k in range(num_seg):
				sig_est[k,:]+=alpha_est[j]*computeKern(t_obs-cen_cv[j]-k*T[i],width_est[j])

		Q[i]=np.sum((Data-sig_est)**2/noise_std**2)

	return T[np.argmin(Q)]






def computeKern(tau,sigma):
	'''
	Smoothing kernel
	'''
	return np.exp(-tau*tau*(1.0/(sigma*sigma)))

def compute_K(t_obs,cent,width,delay,num_seg):

	'''
	This function computes the matrix K used in the kernel method described in 
	Cuevas-Tello JC, Tino P, Raychaudhury S (2006) "How accurate are the time delay estimates in gravitational
	lensing?" Astronomy and Astrophysics 454: 695-706.
	'''

	a1=t_obs[:,None]-cent[None,:]
	a2=t_obs[:,None]-cent[None,:]-delay
	K=np.concatenate((np.exp(-a1**2/width**2),np.exp(-a2**2/width**2)))
	

	return K

def gen_data_noconv(t_obs,num_mix,delay,num_seg):

	'''
	This function generates artificial data from
	a mixture of Gaussian kernels
	'''

	

	
	sig=np.zeros((num_seg,len(t_obs)))
	cen=np.zeros(num_mix)
	width=np.zeros(num_mix)
	beta=np.zeros(num_mix)
	#get center and width of the Gaussian

	for i in range(num_mix):
	
		cen[i]=np.random.rand(1)[0]*t_obs[-1]*0.25+2.5#the Gaussian centers are in the interval [2.5,5]
		width[i]=np.random.rand(1)[0]+0.5
		beta[i]=np.random.rand(1)[0]
		for j in range(num_seg):
			sig[j,:]+=(beta[i]*np.exp(-(t_obs-cen[i]-j*delay)*(t_obs-cen[i]-j*delay)*(1.0/(width[i]*width[i]))))
		
	

	return sig

def gen_data_conv(t_obs,num_mix,delay,num_seg):

	'''
	This function generates artificial data from
	a mixture of Gaussian kernels and convolves the result with a Gaussian kernel.

	The convolution can be computed analytically
	'''

	

	
	sig=np.zeros((num_seg,len(t_obs)))
	cen=np.zeros(num_mix)
	width=np.zeros(num_mix)
	beta=np.zeros(num_mix)
	
	#generate the parameters for the latent function
	for i in range(num_mix):
		cen[i]=np.random.rand(1)[0]*t_obs[-1]*0.25+2.5#the Gaussian centers are in the interval [2.5,5]
		width[i]=np.random.rand(1)[0]+0.5
		beta[i]=np.random.rand(1)[0]

	#get center and width of the Gaussian
	for i in range(num_seg):
		li=np.random.rand(1)[0]*t_obs[-1]*.1875+t_obs[-1]*.0625#Maximum li is 2.5
		alpha=np.random.rand(1)[0]
		for j in range(num_mix):
			sig[i,:]+=(alpha*beta[j]*(1.0/(np.sqrt(2*li*li+width[j]*width[j])))*np.exp(-0.5*(t_obs-cen[j]-i*delay)*(t_obs-cen[j]-i*delay)*(1.0/(0.5*width[j]*width[j]+li*li))))#Analytical result of the convolution of two Gaussians
			
	
	

	return sig

def upperbound_toy(num_seg):

	'''
	This function defines the bounds for the parameters
	'''

	x=np.zeros(1+3*(num_seg))
	
	#l_f
	x[0]=10.0

	#alpha
	ind=1
	for i in range(ind,ind+num_seg):
		x[i]=1.0
	
	#Delay
	ind=ind+num_seg
	j=1
	for i in range(ind,ind+num_seg-1):
		x[i]=2*num_seg
		j+=1
	
	#l
	ind=ind+num_seg-1
	for i in range(ind,ind+num_seg):
		x[i]=1.0
	#noise
	ind=ind+num_seg
	for i in range(ind,len(x)):
		x[i]=1.0
	return x

def upperbound_delay(num_seg):

	'''
	This function defines the bounds for the parameters
	'''

	x=np.zeros(1+2*(num_seg))
	
	#l_f
	x[0]=10.0

	#alpha
	ind=1
	for i in range(ind,ind+num_seg):
		x[i]=1.0
	
	#Delay
	ind=ind+num_seg
	j=1
	for i in range(ind,ind+num_seg-1):
		x[i]=2*num_seg
		j+=1
	
	
	#noise
	ind=ind+num_seg-1
	for i in range(ind,len(x)):
		x[i]=1.0
	return x



if len(sys.argv) != 4:  
	sys.exit("Usage: run Toy_data.py [conv] [num_obs] [num_trials]")

conv=int(sys.argv[1]) #When conv is 1 the toy data is distorted via convolution
num_obs=int(sys.argv[2]) #Number of observations
num_trials=int(sys.argv[3])#Number of random data initializations
start=0 # Start of observation interval
stop=10 # End of observation interval
num_mix=20 # Number of mixtures in synthetic data
num_seg=2 # Number of data streams. We estimate the delay between two data streams
num_bins=100#Number of bins in DCF method




#square error for each technique
err_gp_noconv=np.zeros(num_trials)
err_gp_conv=np.zeros(num_trials)
err_dcf=np.zeros(num_trials)
err_xcorr=np.zeros(num_trials)
err_kern=np.zeros(num_trials)

np.random.seed(123)


	
#equal spacing
delta=float(stop)/(num_obs-1)
t_obs=np.arange(num_obs)*delta

print '#######################'
print '# Experiments Running #'
print '#######################'



for i in range(0,num_trials): 

	print 'Num Obs ',num_obs,' Trial: ',i

	D=0.15*stop*np.random.rand()+1 #Delay in interval [1,2.5]
	noise_std=0.001

	#Generate the toy data
	
	if conv:
		Data=gen_data_conv(t_obs,num_mix,D,num_seg)
	else:
		Data=gen_data_noconv(t_obs,num_mix,D,num_seg)
	
	
	noise=np.random.multivariate_normal(np.zeros(num_obs),noise_std*noise_std*np.eye(num_obs))
	Y=[]
	
	for j in range(0,num_seg):
		
		Y=np.concatenate((Y,Data[j,:]+noise))
		
	Data+=noise
	

	#Estimate delay using DCF
	err_dcf[i]=np.square(D-dcf(Data[0,:],Data[1,:],t_obs,num_bins,noise_std)[0])/np.square(D)
	#Estimate delay using Cross correlation
	xcorr=np.correlate(Data[1,:],Data[0,:],"full")
	D_est_xcorr=(np.argmax(xcorr)-(len(Data[0,:])-1))*np.diff(t_obs)[0]
	err_xcorr[i]=np.square(D-D_est_xcorr)/np.square(D)

	#Estimate delay using Kern
	err_kern[i]=np.square(D-kernDelay(Data,t_obs,50,num_seg,noise_std))/np.square(D)
	

	#Estimate delay using GP-NoConv
	num_param=1+2*(num_seg)
	a=np.zeros(num_param)
	b=upperbound_delay(num_seg)

	diag=0
	trans=1
	bound=10.0
	num_try=10#number of random initiaizations
	opt=np.zeros((num_try,num_param+1))
	
	for j in range(0,num_try):
		x0=np.random.rand(num_param)
		if trans==1:
			x0=cgf.paramTrans(x0,a,b)
		xopt=sp.optimize.fmin_tnc(dgf.loglik_tied_fsf_delay, x0, dgf.grad_loglik_tied_fsf_delay, args=(np.concatenate((t_obs,t_obs)),Y,num_seg,trans,a,b,diag,np.concatenate(([num_obs],[num_obs]))),approx_grad=0, bounds=[(-bound,bound) for k in range(0,len(x0))],messages=0)[0]
		opt[j,:]=np.concatenate((xopt,np.array([ -dgf.loglik_tied_fsf_delay(xopt,np.concatenate((t_obs,t_obs)),Y,num_seg,trans,a,b,diag,np.concatenate(([num_obs],[num_obs])))])))
		
		
	#get the optimum parameters
	xopt=opt[np.argmax(opt[:,num_param]),0:num_param]
	
	if trans==1:
		xopt=cgf.paramInvTrans(xopt,a,b)
		

	D_est=xopt[1+num_seg:2*num_seg]

	err_gp_noconv[i]=np.square(np.linalg.norm(xopt[1+num_seg:2*num_seg]-D*(np.arange(num_seg-1)+1))/np.linalg.norm(D*(np.arange(num_seg-1)+1)))
	#Estimate delay using GP-NoConv
	num_param=1+3*(num_seg)
	a=np.zeros(num_param)
	b=upperbound_toy(num_seg)

	diag=0
	trans=1
	bound=10.0
	num_try=10#number of random initiaizations
	opt=np.zeros((num_try,num_param+1))
	
	for j in range(0,num_try):
		x0=np.random.rand(num_param)
		if trans==1:
			x0=cgf.paramTrans(x0,a,b)
		xopt=sp.optimize.fmin_tnc(cgf.loglik_tied_fsf, x0, cgf.grad_loglik_tied_fsf, args=(t_obs,Y,num_seg,trans,a,b,diag),approx_grad=0, bounds=[(-bound,bound) for k in range(0,len(x0))],messages=0)[0]
		opt[j,:]=np.concatenate((xopt,np.array([ -cgf.loglik_tied_fsf(xopt,t_obs,Y,num_seg,trans,a,b,diag)])))

		

	#get the optimum parameters
	xopt=opt[np.argmax(opt[:,num_param]),0:num_param]
	
	if trans==1:
		xopt=cgf.paramInvTrans(xopt,a,b)
		

	D_est=xopt[1+num_seg:2*num_seg]

	err_gp_conv[i]=np.square(np.linalg.norm(xopt[1+num_seg:2*num_seg]-D*(np.arange(num_seg-1)+1))/np.linalg.norm(D*(np.arange(num_seg-1)+1)))
	


#Save and display result

quant=mquantiles(err_gp_conv, prob=[0.25, 0.5, 0.75])
quant2=mquantiles(err_xcorr, prob=[0.25, 0.5, 0.75])
quant3=mquantiles(err_dcf, prob=[0.25, 0.5, 0.75])
quant4=mquantiles(err_gp_noconv, prob=[0.25, 0.5, 0.75])
quant5=mquantiles(err_kern, prob=[0.25, 0.5, 0.75])



print '#####################'
print '#      Results      #'
print '#####################'
print '%-10s %-10s'%('Method','MNSE')
print '%-10s %-10f'%('Corr', quant2[1])
print '%-10s %-10f'%('DCF', quant3[1])
print '%-10s %-10f'%('Kern',quant5[1])
print '%-10s %-10f'%('GP-NoConv',quant4[1])
print '%-10s %-10f'%('GP-Conv',quant[1])





