# This file contains a number of useful function definitions for implementing the  
# delay estimation using a Gaussian process framework without convolution
#
# Ciira wa Maina, 2014
# Dedan Kimathi University of Technology.
# Nyeri-Kenya


import pylab as pb
import numpy as np
import scipy as sp
from scipy import integrate
from scipy import special
from scipy.optimize import fmin_tnc
import scipy.linalg
import sys


def rbf2(t1,t2,sigma,l):

	'''
	RBF Kernel
	'''

	t3= t1[:,None]-t2[None,:]
	return sigma*sigma*np.exp(-(1/(2.0*l*l))*t3*t3)


def genCov_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,len_seg_obs):

	'''
	Covariance of gene segments profiles and latent function
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((len(t),len(t)))#assume len(t)= len(t1)
	indx=np.concatenate((np.array([0]),np.cumsum(len_seg_obs)))#allow each segment to have different length

	for i in range(0,num_seg):
		for j in range(i,num_seg):
					
			if i==j:
				K[indx[i]:indx[i+1],indx[j]:indx[j+1]]=alpha[i]*alpha[i]*rbf2(t[indx[i]:indx[i+1]],t[indx[i]:indx[i+1]],sigma_rbf,l_rbf)+noise_std[i]*noise_std[i]*np.eye(len_seg_obs[i])
			else:
				K[indx[i]:indx[i+1],indx[j]:indx[j+1]]=alpha[i]*alpha[j]*rbf2(t[indx[i]:indx[i+1]]-D[i],t[indx[j]:indx[j+1]]-D[j],sigma_rbf,l_rbf)

			if i!=j:
				K[indx[j]:indx[j+1],indx[i]:indx[i+1]]=K[indx[i]:indx[i+1],indx[j]:indx[j+1]].T
				
				
	
	return K






def genCov_l_f_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,len_seg_obs):
	'''
	Gradient of Covariance w.r.t l_f
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((len(t),len(t)))#assume len(t)= len(t1)
	indx=np.concatenate((np.array([0]),np.cumsum(len_seg_obs)))#allow each segment to have different length
	for i in range(0,num_seg):
		for j in range(i,num_seg):
			t1=t[indx[i]:indx[i+1]]
			t2=t[indx[j]:indx[j+1]]
			T_d=(t1[:,None]-t2[None,:])
					
			if i==j:
				K[indx[i]:indx[i+1],indx[j]:indx[j+1]]=alpha[i]*alpha[i]*rbf2(t[indx[i]:indx[i+1]],t[indx[i]:indx[i+1]],sigma_rbf,l_rbf)*T_d*T_d*(1/(l_rbf**3))
			else:
				K[indx[i]:indx[i+1],indx[j]:indx[j+1]]=alpha[i]*alpha[j]*rbf2(t[indx[i]:indx[i+1]]-D[i],t[indx[j]:indx[j+1]]-D[j],sigma_rbf,l_rbf)*(T_d+D[j])*(T_d+D[j])*(1/(l_rbf**3))

			if i!=j:
				K[indx[j]:indx[j+1],indx[i]:indx[i+1]]=K[indx[i]:indx[i+1],indx[j]:indx[j+1]].T
				
				
	
	return K





def genCov_alpha_i_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,seg,len_seg_obs):

	'''
	Gradient of Covariance w.r.t alpha_i
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((len(t),len(t)))#assume len(t)= len(t1)
	indx=np.concatenate((np.array([0]),np.cumsum(len_seg_obs)))#allow each segment to have different length

	
	for j in range(0,num_seg):
				
		if seg==j:
			K[indx[seg]:indx[seg+1],indx[j]:indx[j+1]]=alpha[seg]*rbf2(t[indx[seg]:indx[seg+1]]-D[seg],t[indx[seg]:indx[seg+1]]-D[seg],sigma_rbf,l_rbf)
		else:
			K[indx[seg]:indx[seg+1],indx[j]:indx[j+1]]=alpha[j]*rbf2(t[indx[seg]:indx[seg+1]]-D[seg],t[indx[j]:indx[j+1]]-D[j],sigma_rbf,l_rbf)

		
			
				
	
	return K+K.T




def genCov_D_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,seg,len_seg_obs):

	'''
	Gradient of Covariance w.r.t D_i
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((len(t),len(t)))#assume len(t)= len(t1)
	indx=np.concatenate((np.array([0]),np.cumsum(len_seg_obs)))#allow each segment to have different length
	
	
	for j in range(0,num_seg):

		t1=t[indx[seg]:indx[seg+1]]
		t2=t[indx[j]:indx[j+1]]
		T_d=(t1[:,None]-t2[None,:])
				
		if seg==j:
			K[indx[seg]:indx[seg+1],indx[j]:indx[j+1]]=np.zeros((len(t1),len(t2)))
		else:
			K[indx[seg]:indx[seg+1],indx[j]:indx[j+1]]=alpha[seg]*alpha[j]*rbf2(t1-D[seg],t2-D[j],sigma_rbf,l_rbf)*(1.0/(l_rbf*l_rbf))*(T_d-D[seg]+D[j])
			K[indx[j]:indx[j+1],indx[seg]:indx[seg+1]]=alpha[seg]*alpha[j]*rbf2(t2-D[j],t1-D[seg],sigma_rbf,l_rbf)*(1.0/(l_rbf*l_rbf))*(T_d.T-D[seg]+D[j])

		
			
				
	
	return K



def genCov_sigma_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,len_seg_obs):

	'''
	Gradient of Covariance w.r.t sigma_i
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((len(t),len(t)))#assume len(t)= len(t1)
	indx=np.concatenate((np.array([0]),np.cumsum(len_seg_obs)))#allow each segment to have different length
	

	for i in range(0,num_seg):
		K[indx[i]:indx[i+1],indx[i]:indx[i+1]]=2.0*noise_std*np.eye(len_seg_obs[i])
	return K







def loglik_tied_fsf_delay(params,t,Y,num_seg,trans,a,b,diag,len_seg_obs):

	if trans==1:
		params=paramInvTrans(params,a,b)

	#unpack parameters
	sigma_rbf=1.0
	l_rbf=params[0]
	ind=1
	alpha=params[ind:ind+num_seg]
	#initilize Delay
	ind=ind+num_seg
	D=np.zeros(num_seg)
	D[1:num_seg]=params[ind:ind+num_seg-1]
	D[0]=0.0
	#initilize noise
	ind=ind+num_seg-1
	noise_std1=params[ind:len(params)]
	noise_std=np.ones(num_seg)*noise_std1
	
	Cov=genCov_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,len_seg_obs)

	if diag:
		Cov=blk_diag(Cov,num_seg,len(t))#need to change

	try:
		L=np.linalg.cholesky(Cov)
	except np.linalg.LinAlgError:
		return -np.inf
	alpha=sp.linalg.cho_solve((L,1),Y)
	ll=-0.5*np.dot(Y[None,:],alpha[:,None])[0,0]-np.sum(np.log(np.diag(L)))-0.5*Y.size*np.log(2*np.pi)
	return -ll








def grad_loglik_tied_fsf_delay(params,t,Y,num_seg,trans,a,b,diag,len_seg_obs):

	grad=np.zeros(len(params))

	if trans==1:
		params=paramInvTrans(params,a,b)

	#unpack parameters
	sigma_rbf=1.0
	l_rbf=params[0]
	ind=1
	alpha=params[ind:ind+num_seg]
	#initilize Delay
	ind=ind+num_seg
	D=np.zeros(num_seg)
	D[1:num_seg]=params[ind:ind+num_seg-1]
	D[0]=0.0
	ind=ind+num_seg-1
	noise_std1=params[ind:len(params)]
	noise_std=np.ones(num_seg)*noise_std1
	
	Cov=genCov_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,len_seg_obs)
	
	if diag:
		Cov=blk_diag(Cov,num_seg,len(t))

	try:
		L=np.linalg.cholesky(Cov)
		#invCov=np.linalg.inv(Cov)
	except np.linalg.LinAlgError:
		return -np.inf
	alpha_cho=sp.linalg.cho_solve((L,1),Y)[:,None]

	#latent function parameters
	
	gK=genCov_l_f_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,len_seg_obs)
	if diag:
		gK=blk_diag(gK,num_seg,len(t))
	#grad[0]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))
	grad[0]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(sp.linalg.cho_solve((L,1),gK))

	#alpha
	ind=1
	j=0
	for i in range(ind,ind+num_seg):
		gK=genCov_alpha_i_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,j,len_seg_obs)
		if diag:
			gK=blk_diag(gK,num_seg,len(t))
		#grad[i]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))
		grad[i]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(sp.linalg.cho_solve((L,1),gK))
		j+=1
	
	#Delay
	ind=ind+num_seg
	j=1
	for i in range(ind,ind+num_seg-1):
		gK=genCov_D_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,j,len_seg_obs)
		if diag:
			gK=blk_diag(gK,num_seg,len(t))
		#grad[i]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))
		grad[i]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(sp.linalg.cho_solve((L,1),gK))
		j+=1
	
	
	# noise
	ind=ind+num_seg-1
	
	gK=genCov_sigma_delay(t,alpha,D,sigma_rbf,l_rbf,noise_std1,num_seg,len_seg_obs)
	if diag:
		gK=blk_diag(gK,num_seg,len(t))
	#grad[ind]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))
	grad[ind]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(sp.linalg.cho_solve((L,1),gK))
	
		
	if trans==1:
		return grad*gradTrans(paramTrans(params,a,b),a,b)
	elif trans==0:
		return grad



def pred_Cov_tied_fsf_delay(t_obs,t_pred,Y,params,num_seg,seg,trans,a,b,len_seg_obs):

	if trans==1:
		params=paramInvTrans(params,a,b)

	indx=np.concatenate((np.array([0]),np.cumsum(len_seg_obs)))#allow each segment to have different length

	#unpack parameters
	sigma_rbf=1.0
	l_rbf=params[0]
	ind=1
	alpha=params[ind:ind+num_seg]
	#initilize Delay
	ind=ind+num_seg
	D=np.zeros(num_seg)
	D[1:num_seg]=params[ind:ind+num_seg-1]
	D[0]=0.0
	ind=ind+num_seg-1
	noise_std1=params[ind:len(params)]
	noise_std=np.ones(num_seg)*noise_std1
	
	
	
	B=genCov_delay(t_obs,alpha,D,sigma_rbf,l_rbf,noise_std,num_seg,len_seg_obs)
	

	A=alpha[seg]*alpha[seg]*rbf2(t_pred,t_pred,sigma_rbf,l_rbf)+noise_std1**2*np.eye(t_pred.size)
	


	C=np.zeros((t_pred.size,t_obs.size))

	
	
	for i in range(0,num_seg):
		t1=t_obs[indx[i]:indx[i+1]]

		CC=np.zeros((t_pred.size,t1.size))
		for j in range(0,t_pred.size):
			for k in range(0,t1.size):
				if t_pred[j]==t1[k]:
					CC[j,k]=1.0

		if i==seg:
			C[0:len(t_pred),indx[i]:indx[i+1]]=alpha[i]*alpha[seg]*rbf2(t_pred-D[seg],t1-D[i],sigma_rbf,l_rbf)+noise_std[seg]*noise_std[seg]*CC
		else:
			C[0:len(t_pred),indx[i]:indx[i+1]]=alpha[i]*alpha[seg]*rbf2(t_pred-D[seg],t1-D[i],sigma_rbf,l_rbf)

	mu=np.dot(C,np.dot(np.linalg.inv(B),Y[:,None]))
	Cov=A-np.dot(C,np.dot(np.linalg.inv(B),C.T))+1e-8*np.eye(t_pred.size)

			




	return {'Cov':Cov,'mu':mu}




def pred_Lat_tied_fsf(t_obs,t_pred,Y,params,num_seg,trans,a,b):

	if trans==1:
		params=paramInvTrans(params,a,b)

	#unpack parameters
	sigma_rbf=1.0
	l_rbf=params[0]
	ind=1
	alpha=params[ind:ind+num_seg]
	#initilize Delay
	ind=ind+num_seg
	D=np.zeros(num_seg)
	D[1:num_seg]=params[ind:ind+num_seg-1]
	D[0]=0.0
	ind=ind+num_seg-1
	#initilize l
	l=params[ind:ind+num_seg]
	#initilize noise
	ind=ind+num_seg
	noise_std1=params[ind:len(params)]
	noise_std=np.ones(num_seg)*noise_std1
	
	B=genCov(t_obs,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg)
	A=rbf2(t_pred,t_pred,sigma_rbf,l_rbf)
	


	C=np.zeros((t_pred.size,t_obs.size*num_seg))

	CC=np.zeros((t_pred.size,t_obs.size))
	for i in range(0,t_pred.size):
		for j in range(0,t_obs.size):
			if t_pred[i]==t_obs[j]:
				CC[i,j]=1.0
	
	for i in range(0,num_seg):
		
		C[0:len(t_pred),i*len(t_obs):(i+1)*len(t_obs)]=alpha[i]*cov_yiLat(t_pred,t_obs,D[i],sigma_rbf,l_rbf,l[i])

	mu=np.dot(C,np.dot(np.linalg.inv(B),Y[:,None]))
	Cov=A-np.dot(C,np.dot(np.linalg.inv(B),C.T))+1e-8*np.eye(t_pred.size)

			




	return {'Cov':Cov,'mu':mu}



def pred_Lat_tied_fsf_new(t_obs,t_pred,Y,params,num_seg,trans,a,b,len_seg_obs):

	if trans==1:
		params=paramInvTrans(params,a,b)

	indx=np.concatenate((np.array([0]),np.cumsum(len_seg_obs)))#allow each segment to have different length

	#unpack parameters
	sigma_rbf=1.0
	l_rbf=params[0]
	ind=1
	alpha=params[ind:ind+num_seg]
	#initilize Delay
	ind=ind+num_seg
	D=np.zeros(num_seg)
	D[1:num_seg]=params[ind:ind+num_seg-1]
	D[0]=0.0
	ind=ind+num_seg-1
	#initilize l
	l=params[ind:ind+num_seg]
	#initilize noise
	ind=ind+num_seg
	noise_std1=params[ind:len(params)]
	noise_std=np.ones(num_seg)*noise_std1
	
	B=genCov_new(t_obs,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg,len_seg_obs)
	A=rbf2(t_pred,t_pred,sigma_rbf,l_rbf)
	


	C=np.zeros((t_pred.size,t_obs.size))

	
	
	for i in range(0,num_seg):
		t1=t_obs[indx[i]:indx[i+1]]

		
		
		C[0:len(t_pred),indx[i]:indx[i+1]]=alpha[i]*cov_yiLat(t_pred,t1,D[i],sigma_rbf,l_rbf,l[i])

	mu=np.dot(C,np.dot(np.linalg.inv(B),Y[:,None]))
	Cov=A-np.dot(C,np.dot(np.linalg.inv(B),C.T))+1e-8*np.eye(t_pred.size)

			




	return {'Cov':Cov,'mu':mu}



def paramTrans(x,a,b):
	return np.log((x-a)/(b-x))
def paramInvTrans(x,a,b):
	return a+((b-a)/(1+np.exp(-x)))
def gradTrans(x,a,b):
	return (((b-a)*np.exp(x))/np.square(1+np.exp(x)))







