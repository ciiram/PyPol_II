OVERVIEW
========

This folder contains Python code used to study the dynamics of RNA polymerase II (RNA Pol-II). 
The associated manuscript is “A Probabilistic Model of Transcription Dynamics applied to Estrogen Signalling in Breast Cancer Cells,” 
by Ciira wa Maina et al. and available as an arxiv preprint here arXiv:1303.4926 [q-bio.QM]. 


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
3. Example input file ACTN1.txt


To run the program open an Ipython shell and type

	run PyPolII.py [-h] [-i INPUT_FILE] [-l GENE_LEN] [-n NUM_TRY] [-t TRANS]
                  [-o OUT_DIR]

or type

	python PyPolII.py [-h] [-i INPUT_FILE] [-l GENE_LEN] [-n NUM_TRY] [-t TRANS]
                  [-o OUT_DIR]

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
		        pol-II segment profiles, <gene name>.pdf, and a text
		        file with the delays of each segment <gene
		        name_delay>.txt





EXAMPLE
=======


Executing 

	run PyPolII.py -i ACTN1.txt -l 105244

will run the model using data of pol-II occupancy for the ACTN1 gene for 5 segments 
and compute the delays of the segments. A figure of the inferred profiles and a file 
with the delays is generated. Using these delays, the transcription speed can be
computed using a linear regression through the origin as described in the paper.  


Citation
========

If you use this program please cite

C. wa Maina, F. Matarese, K. Grote, H. G. Stunnenberg, G. Reid, A. Honkela, N. D. Lawrence, and M. Rattray,
“A Probabilistic Model of Transcription Dynamics applied to Estrogen Signalling in Breast Cancer Cells,”
arXiv:1303.4926 [q-bio.QM]. 




