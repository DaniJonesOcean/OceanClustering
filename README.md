# OceanClustering
Unsupervised Clustering of Ocean Data in python

Readme for GMM code:

The combined program consists of 8 modules.
- Main.py is the central script and determines the values of all the parameters to be used and which other scripts are called during a particular run. The file locations for the input data and output files are specified here.
- Load.py loads, cleans, sub-samples and standardises the data for the rest of the program.
- PCA.py both creates and applies the principal component analysis to the dataset, which is necessary to increase the computational speed of the program
- GMM.py creates and applies sci-kit learn’s Gaussian Mixture Modelling class.
- Reconstruct.py transforms the results from PCA centred space back to the original, physical space (either centred or uncentred).

- Print.py prints the results of the program to csv files along the way and also has methods which can read these results from the files and return them in forms which can be used by the next module.
- Plot.py uses Print.py to generate plots and maps of the results.
- Bic.py runs more independently from the other scripts and uses BIC scores to determine the ideal number of Gaussian components for the model. 

Library requirements:
- Python 3.5.2
- Scikit-learn 0.18.1
- h5py 2.6.0 (used to import *.mat datafile)
- numpy 1.11.3
- scipy 0.19.0
- matplotlib 1.5.3
- pickle (part of the standard python library)
- Cartopy 0.15.1 (for creating stereographic projection maps)

Assumed file structure:
The program takes three input addresses and then assumes a certain file structure beyond this point. If the directories do not already exist, the program automatically creates them. It does not create the "Data_in" or "Fronts" directories, though. It only creates the ones listed below.
- address = location for storing the outputs of the program
- filename_raw_data = location of the raw data in a .mat file
- address_fronts = location of the front data in .txt files
The assumed file structure is all within the “address” file:
Address
- Code
- Data_store
	- CentredAndUncentred
	- CentredAndUncentred_Train
	- CentredAndUncentred_Test
	- GMM_classes_depth
	- GMM_classes_reduced
	- GMM_classes_uncentred
	- Info
	- Labels
	- PCA
	- PCA_Train
	- Probabilities
	- Reconstruction
	- Reconstruction_Train
- Objects
- Plots
- Results
