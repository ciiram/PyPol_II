
# This program computes the delay between the different gene segments using the   
# convolved Gaussian process framework for RNA pol-II dynamics
#
# Ciira wa Maina, 2013
# Dedan Kimathi University of Technology.
# Nyeri-Kenya

import sys
import numpy as np
import pylab as pb
import conv_gp_funcs as cgf
import scipy as sp
from scipy.optimize import fmin_tnc
import argparse



#Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', dest='input_file', default='ACTN1.txt',help='Properly Formatted Input file. It is assumed that the file name is in the form <gene name>.txt')
parser.add_argument('-l', '--gene_length', dest='gene_len',type=float, help='Gene length')
parser.add_argument('-n', '--num_try', dest='num_try', type=int,default=1,help='Number of random initializations when performing maximum likelihood optimization')
parser.add_argument('-t', '--trans', dest='trans', type=bool,default='True',help='Parameter transformation flag. When true, the parameters are transformed using a logit function before optimization.')
parser.add_argument('-o', '--out_dir', dest='out_dir', default='',help='The complete path of the output directory to store program output. The outputs are a plot of the inferred pol-II segment profiles, <gene name>.pdf, and a text file with the delays of each segment <gene name_delay>.txt. If not supplied the outputs are stored in the current directory.')
args = parser.parse_args()


Data=np.genfromtxt(args.input_file)#Load the properly formated data
obs_time=Data[:,0]#Extract the observation times
num_obs=len(obs_time)#Number of observations
num_seg=Data.shape[1]-1#Number of gene segments or observation streams
per=1.0/num_seg#percentage of gene corresponding to each segment
num_param=1+3*(num_seg) #number of parameters in the model
bound=10.0#Bound on the transformed variable to prevent numerical instability
#Obtain the parameter bounds
a=cgf.lowerbound_tied_fsf(num_seg,args.gene_len,per)
b= cgf.upperbound_tied_fsf(num_seg,args.gene_len,per)
bound=10.0#Bound on the transformed variable to prevent numerical instability
gene=args.input_file.split('.')[0]


#Form a vector of the observed time series
Y=[]
for i in range(0,num_seg):
	Y=np.concatenate((Y,Data[0:num_obs,i+1]))

opt=np.zeros((args.num_try,num_param+1))#store the final parameters and final loglikelihood
diag=0
for i in range(0,args.num_try):
	
	#we try a number of random initializations and chose the one leading to maximum loglikelihood
	x0=cgf.init_param_tied_fsf(num_seg,args.trans,a,b,args.gene_len,per)
	
	if args.trans==1:
		xopt=sp.optimize.fmin_tnc(cgf.loglik_tied_fsf, x0, cgf.grad_loglik_tied_fsf, args=(obs_time,Y,num_seg,args.trans,a,b,diag),approx_grad=0, bounds=[(-bound,bound) for j in range(0,len(x0))],messages=0)[0]
	
	opt[i,:]=np.concatenate((xopt,np.array([ -cgf.loglik_tied_fsf(xopt,obs_time,Y,num_seg,args.trans,a,b,diag)])))
#get the optimum parameters
xopt=opt[np.argmax(opt[:,num_param]),0:num_param]


#make some predictions
t_pred=np.linspace(obs_time[0],obs_time[len(obs_time)-1],500)#prediction times
#t_pred=np.linspace(obs_time[0],160.0,500)#prediction times


pb.figure()
seg_color=['b','g','c','r','m','y','b','g','c','r','m','y']
seg_mark=['o','s','d','p','*','>','o','s','d','p','*','>']
ymax=np.ceil(np.max(Data))
yy=np.linspace(0,1,np.ceil(1/per)+1)*100
pb.subplot(num_seg+1,1,1)
Res=cgf.pred_Lat_tied_fsf(obs_time,t_pred,Y,xopt,num_seg,args.trans,a,b)
pb.plot(t_pred,Res['mu'],seg_color[0],linewidth=2)
mu=Res['mu']
Cov=Res['Cov']
pb.plot(t_pred,mu[:,0]+2*np.sqrt(np.diag(Cov)),'--'+seg_color[0])
pb.plot(t_pred,mu[:,0]-2*np.sqrt(np.diag(Cov)),'--'+seg_color[0])
pb.yticks([])
pb.xticks([])
pb.title("Pol-II activity over different segments of the "+gene+" gene ")

for i in range(0,num_seg):
	pb.subplot(num_seg+1,1,i+2)

	pb.plot(obs_time,Data[0:num_obs,i+1],seg_color[i+1]+seg_mark[i+1])
	Res=cgf.pred_Cov_tied_fsf(obs_time,t_pred,Y,xopt,num_seg,i,args.trans,a,b)
	pb.plot(t_pred,Res['mu'],seg_color[i+1],linewidth=2)
	mu=Res['mu']
	Cov=Res['Cov']
	pb.plot(t_pred,mu[:,0]+2*np.sqrt(np.diag(Cov)),'--'+seg_color[i+1])
	pb.plot(t_pred,mu[:,0]-2*np.sqrt(np.diag(Cov)),'--'+seg_color[i+1])
	pb.yticks([])
	if i!=num_seg-1:
		pb.xticks([])
	pb.ylabel((str(int(yy[i]))+'-'+str(int(yy[i+1]))+'%'))
pb.xlabel("Time (min)")	
figname=args.out_dir+gene+'.pdf'
pb.savefig(figname)
#pb.show()


if args.trans==1:
	xopt=cgf.paramInvTrans(xopt,a,b)
#Obtain the delay parameters
ind=1+num_seg
D=xopt[ind:ind+num_seg-1]
np.savetxt(args.out_dir+gene+'_delay.txt',D)








