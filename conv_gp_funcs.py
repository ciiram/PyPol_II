# This file contains a number of useful function definitions for implementing the  
# convolved Gaussian process framework for RNA pol-II dynamics
#
# Ciira wa Maina, 2013
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



def computeGaussian(tau,sigma):
	'''
	Smoothing kernel
	'''
	return (1.0/(np.sqrt(2*np.pi)*sigma))*np.exp(-tau*tau*(1.0/(2.0*sigma*sigma)))

def rbf(t1,t2,sigma,l):

	'''
	rbf kernel
	'''

	t3= t1-t2
	return sigma*sigma*np.exp(-(1/(2.0*l*l))*t3*t3)

def rbf2(t1,t2,sigma,l):

	t3= t1[:,None]-t2[None,:]
	return sigma*sigma*np.exp(-(1/(2.0*l*l))*t3*t3)




def cov_yiyi(t1,t2,Di,sigma_rbf,l_rbf,l_kern):
	K=np.zeros((len(t1),len(t2)))
	for x in range(0,len(t1)):
		for y in range(0,len(t2)):
			K[x,y]=np.sqrt(2*np.pi)*sigma_rbf*sigma_rbf*l_rbf*computeGaussian(t2[y]-t1[x],np.sqrt(l_rbf*l_rbf+2*l_kern*l_kern))
	return K

def cov_yiyj(t1,t2,Di,Dj,sigma_rbf,l_rbf,li,lj):
	'''
	Covariance of gene segments profiles
	Input:
	Di,Dj are the delays
	sigma_rbf,l_rbf are the latent function parameters
	li,lj variances of the convolution kernel
	'''

	K=np.zeros((len(t1),len(t2)))
	for x in range(0,len(t1)):
		for y in range(0,len(t2)):
			K[x,y]=np.sqrt(2*np.pi)*sigma_rbf*sigma_rbf*l_rbf*computeGaussian(t2[y]-t1[x]+Di-Dj,np.sqrt(l_rbf*l_rbf+li*li+lj*lj))
	return K


def cov_yiLat(t1,t2,Di,sigma_rbf,l_rbf,li):
	'''
	Covariance of gene segments profiles and latent function
	Input:
	Di,Dj are the delays
	sigma_rbf,l_rbf are the latent function parameters
	li,lj variances of the convolution kernel
	'''
	K=np.zeros((len(t1),len(t2)))
	for x in range(0,len(t1)):
		for y in range(0,len(t2)):
			K[x,y]=np.sqrt(2*np.pi)*sigma_rbf*sigma_rbf*l_rbf*computeGaussian(t2[y]-t1[x]-Di,np.sqrt(l_rbf*l_rbf+li*li))
	return K

def genCov(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg):

	'''
	Covariance of gene segments profiles and latent function
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((num_seg*len(t),num_seg*len(t)))#assume len(t)= len(t1)

	for i in range(0,num_seg):
		for j in range(i,num_seg):
					
			if i==j:
				K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)]=alpha[i]*alpha[i]*cov_yiyi(t,t,D[i],sigma_rbf,l_rbf,l[i])+noise_std[i]*noise_std[i]*np.eye(t.size)
			else:
				K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)]=alpha[i]*alpha[j]*cov_yiyj(t,t,D[i],D[j],sigma_rbf,l_rbf,l[i],l[j])

			if i!=j:
				K[j*len(t):(j+1)*len(t),i*len(t):(i+1)*len(t)]=K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)].T
				
				
	
	return K

def genCov_sigma_f(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg):

	'''
	Gradient of Covariance w.r.t sigma_f
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((num_seg*len(t),num_seg*len(t)))

	for i in range(0,num_seg):
		for j in range(i,num_seg):
					
			if i==j:
				K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)]=2.0*(1.0/sigma_rbf)*alpha[i]*alpha[i]*cov_yiyi(t,t,D[i],sigma_rbf,l_rbf,l[i])
			else:
				K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)]=2.0*(1.0/sigma_rbf)*alpha[i]*alpha[j]*cov_yiyj(t,t,D[i],D[j],sigma_rbf,l_rbf,l[i],l[j])

			if i!=j:
				K[j*len(t):(j+1)*len(t),i*len(t):(i+1)*len(t)]=K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)].T
				
				
	
	return K

def genCov_l_f(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg):
	'''
	Gradient of Covariance w.r.t l_f
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((num_seg*len(t),num_seg*len(t)))#assume len(t)= len(t1)
	T_d=-(t[:,None]-t[None,:])
	for i in range(0,num_seg):
		for j in range(i,num_seg):
					
			if i==j:
				K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)]=alpha[i]*alpha[i]*(1.0/l_rbf)*((l[i]*l[i]+l[j]*l[j])/(l_rbf*l_rbf+l[i]*l[i]+l[j]*l[j]))*cov_yiyi(t,t,D[i],sigma_rbf,l_rbf,l[i])+alpha[i]*alpha[i]*((l_rbf)/np.square(l_rbf*l_rbf+l[i]*l[i]+l[j]*l[j]))*cov_yiyi(t,t,D[i],sigma_rbf,l_rbf,l[i])*(T_d+D[i]-D[j])*(T_d+D[i]-D[j])
			else:
				K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)]=alpha[i]*alpha[j]*(1.0/l_rbf)*((l[i]*l[i]+l[j]*l[j])/(l_rbf*l_rbf+l[i]*l[i]+l[j]*l[j]))*cov_yiyj(t,t,D[i],D[j],sigma_rbf,l_rbf,l[i],l[j])+alpha[i]*alpha[j]*cov_yiyj(t,t,D[i],D[j],sigma_rbf,l_rbf,l[i],l[j])*((l_rbf)/np.square(l_rbf*l_rbf+l[i]*l[i]+l[j]*l[j]))*(T_d+D[i]-D[j])*(T_d+D[i]-D[j])

			if i!=j:
				K[j*len(t):(j+1)*len(t),i*len(t):(i+1)*len(t)]=K[i*len(t):(i+1)*len(t),j*len(t):(j+1)*len(t)].T
				
				
	
	return K

def genCov_alpha_i(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg,seg):

	'''
	Gradient of Covariance w.r.t alpha_i
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((num_seg*len(t),num_seg*len(t)))

	
	for j in range(0,num_seg):
				
		if seg==j:
			K[seg*len(t):(seg+1)*len(t),j*len(t):(j+1)*len(t)]=alpha[seg]*cov_yiyi(t,t,D[seg],sigma_rbf,l_rbf,l[seg])
		else:
			K[seg*len(t):(seg+1)*len(t),j*len(t):(j+1)*len(t)]=alpha[j]*cov_yiyj(t,t,D[seg],D[j],sigma_rbf,l_rbf,l[seg],l[j])

		
			
				
	
	return K+K.T

def genCov_D(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg,seg):

	'''
	Gradient of Covariance w.r.t D_i
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((num_seg*len(t),num_seg*len(t)))
	T_d=-(t[:,None]-t[None,:])
	
	for j in range(0,num_seg):
				
		if seg==j:
			K[seg*len(t):(seg+1)*len(t),j*len(t):(j+1)*len(t)]=np.zeros((len(t),len(t)))
		else:
			K[seg*len(t):(seg+1)*len(t),j*len(t):(j+1)*len(t)]=-alpha[seg]*alpha[j]*cov_yiyj(t,t,D[seg],D[j],sigma_rbf,l_rbf,l[seg],l[j])*(1.0/(l_rbf*l_rbf+l[seg]*l[seg]+l[j]*l[j]))*(T_d+D[seg]-D[j])
			K[j*len(t):(j+1)*len(t),seg*len(t):(seg+1)*len(t)]=alpha[seg]*alpha[j]*cov_yiyj(t,t,D[j],D[seg],sigma_rbf,l_rbf,l[j],l[seg])*(1.0/(l_rbf*l_rbf+l[seg]*l[seg]+l[j]*l[j]))*(T_d-D[seg]+D[j])

		
			
				
	
	return K


def genCov_l_i(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg,seg):

	'''
	Gradient of Covariance w.r.t l_i
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((num_seg*len(t),num_seg*len(t)))
	T_d=-(t[:,None]-t[None,:])

	
	for j in range(0,num_seg):
				
		K[seg*len(t):(seg+1)*len(t),j*len(t):(j+1)*len(t)]=-alpha[seg]*alpha[j]*l[seg]*((1.0)/(l_rbf*l_rbf+l[seg]*l[seg]+l[j]*l[j]))*cov_yiyj(t,t,D[seg],D[j],sigma_rbf,l_rbf,l[seg],l[j])+alpha[seg]*alpha[j]*cov_yiyj(t,t,D[seg],D[j],sigma_rbf,l_rbf,l[seg],l[j])*((l[seg])/np.square(l_rbf*l_rbf+l[seg]*l[seg]+l[j]*l[j]))*(T_d+D[seg]-D[j])*(T_d+D[seg]-D[j])

		
			
				
	
	return K+K.T

def genCov_sigma_i(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg,seg):

	'''
	Gradient of Covariance w.r.t sigma_i
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((num_seg*len(t),num_seg*len(t)))
	T_d=-(t[:,None]-t[None,:])

	K[seg*len(t):(seg+1)*len(t),seg*len(t):(seg+1)*len(t)]=2.0*noise_std[seg]*np.eye(len(t))

		
			
				
	
	return K

def genCov_sigma(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg):

	'''
	Gradient of Covariance w.r.t sigma_i
	Input:
	D: are the delays
	sigma_rbf,l_rbf are the latent function parameters
	l: variances of the convolution kernel
	noise_std: noise variances
	'''

	K=np.zeros((num_seg*len(t),num_seg*len(t)))
	T_d=-(t[:,None]-t[None,:])

	for i in range(0,num_seg):
		K[i*len(t):(i+1)*len(t),i*len(t):(i+1)*len(t)]=2.0*noise_std*np.eye(len(t))
	return K




def loglik_tied_fsf(params,t,Y,num_seg,trans,a,b,diag):

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
	l=params[ind:ind+num_seg]
	#initilize noise
	ind=ind+num_seg
	noise_std1=params[ind:len(params)]
	noise_std=np.ones(num_seg)*noise_std1
	
	Cov=genCov(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg)

	if diag:
		Cov=blk_diag(Cov,num_seg,len(t))

	try:
		L=np.linalg.cholesky(Cov)
	except np.linalg.LinAlgError:
		return -np.inf
	alpha=sp.linalg.cho_solve((L,1),Y)
	ll=-0.5*np.dot(Y[None,:],alpha[:,None])[0,0]-np.sum(np.log(np.diag(L)))-0.5*Y.size*np.log(2*np.pi)
	return -ll






def grad_loglik_tied_fsf(params,t,Y,num_seg,trans,a,b,diag):

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
	#initilize l
	ind=ind+num_seg-1
	l=params[ind:ind+num_seg]
	#initilize noise
	ind=ind+num_seg
	noise_std1=params[ind:len(params)]
	noise_std=np.ones(num_seg)*noise_std1
	
	Cov=genCov(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg)
	
	if diag:
		Cov=blk_diag(Cov,num_seg,len(t))

	try:
		L=np.linalg.cholesky(Cov)
		invCov=np.linalg.inv(Cov)
	except np.linalg.LinAlgError:
		return -np.inf
	alpha_cho=sp.linalg.cho_solve((L,1),Y)[:,None]

	#latent function parameters
	
	gK=genCov_l_f(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg)
	if diag:
		gK=blk_diag(gK,num_seg,len(t))
	grad[0]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))

	#alpha
	ind=1
	j=0
	for i in range(ind,ind+num_seg):
		gK=genCov_alpha_i(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg,j)
		if diag:
			gK=blk_diag(gK,num_seg,len(t))
		grad[i]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))
		j+=1
	
	#Delay
	ind=ind+num_seg
	j=1
	for i in range(ind,ind+num_seg-1):
		gK=genCov_D(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg,j)
		if diag:
			gK=blk_diag(gK,num_seg,len(t))
		grad[i]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))
		j+=1
	# l
	ind=ind+num_seg-1
	j=0
	for i in range(ind,ind+num_seg):
		gK=genCov_l_i(t,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg,j)
		if diag:
			gK=blk_diag(gK,num_seg,len(t))
		grad[i]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))
		j+=1
	
	# noise
	ind=ind+num_seg
	
	gK=genCov_sigma(t,alpha,D,sigma_rbf,l_rbf,l,noise_std1,num_seg)
	if diag:
		gK=blk_diag(gK,num_seg,len(t))
	grad[ind]=-0.5*np.dot(alpha_cho.T,np.dot(gK,alpha_cho))[0,0]+0.5*np.trace(np.dot(invCov,gK))
	
		
	if trans==1:
		return grad*gradTrans(paramTrans(params,a,b),a,b)
	elif trans==0:
		return grad





def pred_Cov_tied_fsf(t_obs,t_pred,Y,params,num_seg,seg,trans,a,b):

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
	l=params[ind:ind+num_seg]
	#initilize noise
	ind=ind+num_seg
	noise_std1=params[ind:len(params)]
	noise_std=np.ones(num_seg)*noise_std1
	
	B=genCov(t_obs,alpha,D,sigma_rbf,l_rbf,l,noise_std,num_seg)
	A=alpha[seg]*alpha[seg]*cov_yiyi(t_pred,t_pred,D[seg],sigma_rbf,l_rbf,l[seg])+noise_std[seg]*noise_std[seg]*np.eye(t_pred.size)
	


	C=np.zeros((t_pred.size,t_obs.size*num_seg))

	CC=np.zeros((t_pred.size,t_obs.size))
	for i in range(0,t_pred.size):
		for j in range(0,t_obs.size):
			if t_pred[i]==t_obs[j]:
				CC[i,j]=1.0
	
	for i in range(0,num_seg):
		
		if i==seg:
			C[0:len(t_pred),i*len(t_obs):(i+1)*len(t_obs)]=alpha[i]*alpha[seg]*cov_yiyj(t_pred,t_obs,D[seg],D[i],sigma_rbf,l_rbf,l[seg],l[i])+noise_std[seg]*noise_std[seg]*CC
		else:
			C[0:len(t_pred),i*len(t_obs):(i+1)*len(t_obs)]=alpha[i]*alpha[seg]*cov_yiyj(t_pred,t_obs,D[seg],D[i],sigma_rbf,l_rbf,l[seg],l[i])

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


def func(x,t,Y,num_seg,trans,a,b):
	
	return -loglik3(t,Y,x,num_seg,trans,a,b)
def gaussian_pdf(x,mu,sigma):
	return (1.0/(np.sqrt(2.0*np.pi)*sigma))*np.exp(-(1.0/(2.0*sigma*sigma))*np.square(x-mu))

def paramTrans(x,a,b):
	return np.log((x-a)/(b-x))
def paramInvTrans(x,a,b):
	return a+((b-a)/(1+np.exp(-x)))
def gradTrans(x,a,b):
	return (((b-a)*np.exp(x))/np.square(1+np.exp(x)))
def transDist(x,mu,sigma,a,b):
	return np.abs(1.0/(gradTrans(paramTrans(x,a,b),a,b)))*gaussian_pdf(paramTrans(x,a,b),mu,sigma)

def hmc_func_tied_fsf(params,t,Y,num_seg,trans,pvar,a,b,diag):
	return loglik_tied_fsf(params,t,Y,num_seg,trans,a,b,diag)+(1.0/(2.0*pvar))*np.sum(params*params)
def hmc_fprime_tied_fsf(params,t,Y,num_seg,trans,pvar,a,b,diag):
	return grad_loglik_tied_fsf(params,t,Y,num_seg,trans,a,b,diag)+(1.0/pvar)*params





def init_param_tied_fsf(num_seg,trans,a,b,gene_len,per):

	x=np.zeros(1+3*(num_seg))
	
	lengths=np.array([10.0,20.0,30.0,40.0,80.0])
	p=np.ones(len(lengths))*(1.0/len(lengths))
	smin=500
	smax=5000
	lambda1=np.random.rand(1)[0]
	D=((per*gene_len)/smax*lambda1+(per*gene_len)/smin*(1-lambda1))
	

	
	x[0]=lengths[np.argmax(np.random.multinomial(1,p))]#l_f

	#initialize alpha
	ind=1
	for i in range(ind,ind+num_seg):
		x[i]=np.random.rand(1)[0]
	
	#initilize Delay
	ind=ind+num_seg
	j=1
	for i in range(ind,ind+num_seg-1):
		x[i]=j*D
		j+=1
	ind=ind+num_seg-1
	for i in range(ind,ind+num_seg):
		x[i]=lengths[np.argmax(np.random.multinomial(1,p))]
	#initialize noise
	ind=ind+num_seg
	for i in range(ind,len(x)):
		x[i]=np.random.rand(1)[0]*100
	
	if trans==1:
		return paramTrans(x,a,b)
	elif trans==0:
		return x

def lowerbound_tied_fsf(num_seg,gene_len,per):

	smin=50
	smax=50000

	x=np.zeros(1+3*(num_seg))
	
	
	x[0]=5.0#l_f

	#initialize alpha
	ind=1
	for i in range(ind,ind+num_seg):
		x[i]=0.0
	
	#initilize Delay
	ind=ind+num_seg
	j=1
	for i in range(ind,ind+num_seg-1):
		x[i]=(j*per*gene_len)/smax
		j+=1
	#initilize l
	ind=ind+num_seg-1
	for i in range(ind,ind+num_seg):
		x[i]=5.0
	#initialize noise
	ind=ind+num_seg
	for i in range(ind,len(x)):
		x[i]=0.0
	return x

def upperbound_tied_fsf(num_seg,gene_len,per):

	smin=50
	smax=50000

	

	x=np.zeros(1+3*(num_seg))
	
	
	x[0]=320.0#l_f

	#initialize alpha
	ind=1
	for i in range(ind,ind+num_seg):
		x[i]=100.0
	
	#initilize Delay
	ind=ind+num_seg
	j=1
	for i in range(ind,ind+num_seg-1):
		x[i]=(j*per*gene_len)/smin
		j+=1
	
	#initilize l
	ind=ind+num_seg-1
	for i in range(ind,ind+num_seg):
		x[i]=320.0
	#initialize noise
	ind=ind+num_seg
	for i in range(ind,len(x)):
		x[i]=100.0
	return x




def geneCovRPM(gene,num_seg,series_loc):
	'''
	This function takes a file with the RPMs for a given gene and
	divides it into a number of regions and computes average RPM
	Input:
	gene: the name of the gene
	summary_percentage

	'''
	import sys
	import createBED as cB

	#Now for each interval corresponding to summary_percentage of the gene compute the summary series
	rpm=np.genfromtxt(series_loc+gene+'.txt')
	num_bins=rpm.shape[0]
	gene_segments=np.linspace(0,1,num_seg+1)*num_bins
	summary=np.zeros((rpm.shape[1],len(gene_segments)-1))
	#We need strand information to determine where the gene starts
	strand=cB.getStr(gene)
	for i in range(0,rpm.shape[1]):
		if strand=='+':
			for j in range(0,len(gene_segments)-1):
				summary[i,j]=np.mean(rpm[int(gene_segments[j]):int(gene_segments[j+1]),i])
		elif strand=='-':
			for j in range(len(gene_segments)-1,0,-1):
				summary[i,len(gene_segments)-1-j]=np.mean(rpm[int(gene_segments[j-1]):int(gene_segments[j]),i])
	
	return summary


def blk_diag(Cov,num_blk,dim_blk):
	'''
	Useful to form the diagonal covariance matrix
	'''

	K=np.zeros((num_blk*dim_blk,num_blk*dim_blk))#assume len(t)= len(t1)

	for i in range(0,num_blk):
		K[i*dim_blk:(i+1)*dim_blk,i*dim_blk:(i+1)*dim_blk]=Cov[i*dim_blk:(i+1)*dim_blk,i*dim_blk:(i+1)*dim_blk]
	

	return K





