OVERVIEW
========

This folder contains Python code used to study the dynamics of RNA polymerase II (RNA Pol-II). 
The associated manuscript is “A Probabilistic Model of Transcription Dynamics applied to Estrogen Signalling in Breast Cancer Cells,” 
by Ciira wa Maina et al. and available as an arxiv preprint here [arXiv:1303.4926](http://arxiv.org/abs/1303.4926). 


REQUIREMENTS
============
The programs require Python 2.7 or later and the following python libraries

1. numpy >= 1.6.1
2. scipy >= 0.9.0
3. pylab


INSTALLATION
============
Download the following files in this folder and place them in a folder of your choice.

1. PyPolII.py
2. conv_gp_funcs.py
3. Example data folder Data/


To run the program open an Ipython shell and type

	run PyPolII.py [-h] -i INPUT_FILE -l GENE_LEN [-n NUM_TRY] [-t TRANS]
                  [-o OUT_DIR] [-s RND_SEED]

or type

	python PyPolII.py [-h] -i INPUT_FILE -l GENE_LEN [-n NUM_TRY] [-t TRANS]
                  [-o OUT_DIR] [-s RND_SEED]

directly in the command line.   

The input arguments are

	-h, --help            show this help message and exit
	-i INPUT_FILE, --input_file INPUT_FILE
		        Properly Formatted Input file. It is assumed that the
		        file name is in the form <gene name>.txt
	-l GENE_LEN, --gene_length GENE_LEN
		        Gene length
	-n NUM_TRY, --num_try NUM_TRY
		        Number of random initializations when performing
		        maximum likelihood optimization
	-t TRANS, --trans TRANS
		        Parameter transformation flag. When true, the
		        parameters are transformed using a logit function
		        before optimization.
	-o OUT_DIR, --out_dir OUT_DIR
		        The complete path of the output directory to store
		        program output. The outputs are a plot of the inferred
		        pol-II segment profiles, <gene name>.pdf, a text file
		        with the delays of each segment <gene name_delay>.txt
		        and a text file with the gene transcription speed in
		        kilobases per second <gene name_speed>.txt. If not
		        supplied the outputs are stored in the current
		        directory.
	-s RND_SEED, --rnd_seed RND_SEED
		        Random Seed






EXAMPLE
=======


Executing 

	run PyPolII.py -i Data/ACTN1.txt -l 105244 -s 123

will run the model using data of pol-II occupancy for the ACTN1 gene for 5 segments 
and compute the delays of the segments. A figure of the inferred profiles, a file 
with the delays and a file with the computed transcription speed are generated. Examples are contained in the Results folder. The transcription speed is computed from the segment delays using a linear regression through the origin as described in the paper. In this case the computed speed is *2.8 kilobases per second*. This result is also displayed on the command line as 
	
	ACTN1 2.8 kilobases per second



Input Format
------------

The input file contains the average RNA Pol-II occupancy in reads per million (RPM) over the different gene segments.
For each gene segment, the time series of pol-II occupancy is stored as a column vector. The example file *ACTN1.txt* 
contains the data 
	
	Time	Segment 1	Segment 2	Segment 3	Segment 4	Segment 5

	 0	3.07168506	1.44841574	1.30610211	1.28456781	1.17783258
	 5	4.50312281	2.27195979	1.36380291      1.29378391    	1.24780128
	 ...
	 320    3.71889119	1.77794631	1.65309992	1.63749413	1.76234051

Where the first column contains the time at which the measurements were taken and the remaining 5 columns are the time
series of pol-II occupancy for the 5 gene segments.


Parallel Execution
==================

To allow for the processing of a large number of genes, the program *ParPyPolII.py* allows parallel execution.
This program requires the *IPython.parallel* module which is dependant on 

1. pyzmq
2. tornado

These can be installed by typing

	sudo pip install pyzmq tornado

To run the code in parallel, we must first start a controller and a number of engines by typing for example

	$ ipcluster start -n 4

which starts 4 engines.
 
We can then run our program in the *Ipython* shell by typing

	run ParPyPolII.py [-h] -i GENE_LIST -d DATA_DIR -o OUT_DIR [-n NUM_TRY]
                     [-t TRANS] [-s RND_SEED]

The input arguments are 

	-h, --help            show this help message and exit
	-i GENE_LIST, --gene_list GENE_LIST
		        Properly formatted input file containing gene names
		        and gene lengths. For each gene, the corresponding
		        data should be in the input data directory with the
		        name <gene name>.txt
	-d DATA_DIR, --data_dir DATA_DIR
		        The complete path of the directory containing properly
		        formatted data.
	-o OUT_DIR, --out_dir OUT_DIR
		        The complete path of the output directory to store
		        program output. The outputs are a plot of the inferred
		        pol-II segment profiles, <gene name>.pdf, a text file
		        with the delays of each segment <gene name_delay>.txt
		        and a text file with the gene transcription speed in
		        kilobases per second <gene name_speed>.txt.
	-n NUM_TRY, --num_try NUM_TRY
		        Number of random initializations when performing
		        maximum likelihood optimization
	-t TRANS, --trans TRANS
		        Parameter transformation flag. When true, the
		        parameters are transformed using a logit function
		        before optimization.
	-s RND_SEED, --rnd_seed RND_SEED
		        Random Seed


The file *GENE_LIST* contains a list of genes to be processed as well as their lengths in base pairs separated by a tab (see *gene_list.txt*). For example

	TPM1	22196
	WDR1	42611

For each of the genes in the list, the input data in the format shown above is located in the data directory *DATA_DIR*.

Executing 
	
	run ParPyPolII.py -i gene_list.txt  -d /home/.../Data/ -o /home/.../Results/  -s 123

will run the model for the genes in *gene_list.txt* on the parallel engines. We assume that the file containing definitions for the convolved Gaussian processes (*conv_gp_funcs.py*) is located in the current working directory. The results are stored in the output directory.

	
	 
Citation
========

If you use this program please cite

C. wa Maina, F. Matarese, K. Grote, H. G. Stunnenberg, G. Reid, A. Honkela, N. D. Lawrence, and M. Rattray,
“A Probabilistic Model of Transcription Dynamics applied to Estrogen Signalling in Breast Cancer Cells,”
[arXiv:1303.4926](http://arxiv.org/abs/1303.4926). 




