# This program computes the delay between the different gene segments using the   
# convolved Gaussian process framework for RNA pol-II dynamics
#
# This code supports parallel computations
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
from IPython.parallel import Client


def conv_gp_model_fit(gene,gene_len,trans,num_try,data_dir,save_dir):

	
	
	'''
	This function processes the time series data and fits GPs. 

	INPUTS:
		gene: The gene name
		gene_len: Gene length in base pairs
		trans: Set to 1 to transform the variable via the logit transform
		num_try: Number of restarts for ML 
		data_dir: Location of the reads per million (RPM) files
		save_dir: Directory to save output results

	RESULT:
		The function returns the Maximum likelihood fit of the parameters
		Plots of the ML fits are stored in save_dir as <gene name>.pdf
		The Delays are stored as <gene name_delay>.txt and the computed 
		Transcription speed is stored in <gene name_speed>.txt.
	'''




	import sys
	sys.path.append('/home/ciira/Documents/Research/PlosSubNew/PyPolII')
	import numpy as np
	import pylab as pb
	import conv_gp_funcs as cgf
	import scipy as sp
	from scipy.optimize import fmin_tnc

	Data=np.genfromtxt(data_dir+gene+'.txt')#Load the properly formated data
	obs_time=Data[:,0]#Extract the observation times
	num_obs=len(obs_time)#Number of observations
	num_seg=Data.shape[1]-1#Number of gene segments or observation streams
	per=1.0/num_seg#percentage of gene corresponding to each segment
	num_param=1+3*(num_seg) #number of parameters in the model
	bound=10.0#Bound on the transformed variable to prevent numerical instability
	#Obtain the parameter bounds
	a=cgf.lowerbound_tied_fsf(num_seg,gene_len,per)
	b= cgf.upperbound_tied_fsf(num_seg,gene_len,per)
	bound=10.0#Bound on the transformed variable to prevent numerical instability

	print 'Processing: ',gene
	
	
	Y=[]
	for i in range(0,num_seg):
		Y=np.concatenate((Y,Data[0:num_obs,i+1]))

	opt=np.zeros((num_try,num_param+1))#store the final parameters and final loglikelihood
	opt_diag=np.zeros((num_try,num_param+1))
	diag=0
	for i in range(0,num_try):
		
		#we try a number of random initializations and chose the one leading to maximum loglikelihood
		x0=cgf.init_param_tied_fsf(num_seg,trans,a,b,gene_len,per)
		
		if trans==1:
			xopt=sp.optimize.fmin_tnc(cgf.loglik_tied_fsf, x0, cgf.grad_loglik_tied_fsf, args=(obs_time,Y,num_seg,trans,a,b,diag),approx_grad=0, bounds=[(-bound,bound) for j in range(0,len(x0))],messages=0)[0]
		
		opt[i,:]=np.concatenate((xopt,np.array([ -cgf.loglik_tied_fsf(xopt,obs_time,Y,num_seg,trans,a,b,diag)])))
	#get the optimum parameters
	xopt=opt[np.argmax(opt[:,num_param]),0:num_param]

	#make some predictions
	t_pred=np.linspace(obs_time[0],obs_time[len(obs_time)-1],500)#prediction times
	
	
	pb.figure()
	seg_color=['b','g','c','r','m','y','b','g','c','r','m','y']
	seg_mark=['o','s','d','p','*','>','o','s','d','p','*','>']
	ymax=np.ceil(np.max(Data))
	yy=np.linspace(0,1,np.ceil(1/per)+1)*100
	pb.subplot(num_seg+1,1,1)
	Res=cgf.pred_Lat_tied_fsf(obs_time,t_pred,Y,xopt,num_seg,trans,a,b)
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
		Res=cgf.pred_Cov_tied_fsf(obs_time,t_pred,Y,xopt,num_seg,i,trans,a,b)
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
	
	
	figname=save_dir+gene+'.pdf'
	pb.savefig(figname)
	

	
	if trans==1:
		xopt=cgf.paramInvTrans(xopt,a,b)

	#Obtain the delay parameters
	ind=1+num_seg
	D=xopt[ind:ind+num_seg-1]
	np.savetxt(save_dir+gene+'_delay.txt',D)
	#Compute the transcription speed by performing a  linear regression through the origin
	lengths_gene=gene_len*per*(np.arange(num_seg-1)+1)
	B=np.ones((len(D),1))
	B[:,0]=D
	TransSpeed=np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),lengths_gene)[0]
	np.savetxt(save_dir+gene+'_speed.txt',np.array([np.round(TransSpeed/1000,2)]),fmt='%3.2f')
	print gene,np.round(TransSpeed/1000,1),'kilobases per second'

	return [D,TransSpeed]





#Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--gene_list', dest='gene_list', required=True,help='Properly formatted input file containing gene names and gene lengths.\n For each gene, the corresponding data should be in the input data directory with the name <gene name>.txt\n')
parser.add_argument('-d', '--data_dir', dest='data_dir', required=True,help='The complete path of the directory containing properly formatted data.')
parser.add_argument('-o', '--out_dir', dest='out_dir', required=True,help='The complete path of the output directory to store program output. The outputs are a plot of the inferred pol-II segment profiles, <gene name>.pdf, a text file with the delays of each segment <gene name_delay>.txt and a text file with the gene transcription speed in kilobases per second <gene name_speed>.txt.')
parser.add_argument('-n', '--num_try', dest='num_try', type=int,default=1,help='Number of random initializations when performing maximum likelihood optimization')
parser.add_argument('-t', '--trans', dest='trans', type=bool,default='True',help='Parameter transformation flag. When true, the parameters are transformed using a logit function before optimization.')
parser.add_argument('-s', '--rnd_seed', dest='rnd_seed',type=int, help='Random Seed')
args = parser.parse_args()


rc = Client()
view=rc[:]
view.block = True
rc.load_balanced_view()
ids=rc.ids
view.execute('import numpy as np')
view.execute('import scipy as sp')
view.execute('import scipy.optimize')
view.execute('import pylab as pb')


#Get the genes and gene lengths
file1=open(args.gene_list,"r")
genes=[]
gene_len=[]
while file1:
	line1=file1.readline()
	s1=line1.split()
	if len(s1)==0:
		break
	
	else:
		genes.append(s1[0])
		gene_len.append(float(s1[1]))	
	
file1.close()

#perform the computation in parallel
	
parallel_result = view.map(conv_gp_model_fit,genes,gene_len,[args.trans for i in range(len(genes))],[args.num_try for i in range(len(genes))],[args.data_dir for i in range(len(genes))],[args.out_dir for i in range(len(genes))])



	
