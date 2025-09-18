# RLEvolution 

This repository provides code (under an MIT Licence) for the paper entitled: "RLevolution: Unravelling the history of genomic instability through deep reinforcement learning" by Yun Feng and Christopher Yau from the University of Oxford.

There are two sets of files for training and testing of model.

## Training:

1. Train/Model.py

This file contains the architecture of the whole Q-learning model and reward function.

2. Train/TrainingData.py
 
This file contains the functions for sampling the states and actions used for training.

3. Train/Main_Train.py
 
This file contains the main function used for training.

## Test model:

1. Test/ExtractTrajectory.py

This file contains the deconvolution for copy number history by RLEvolution as well as two heuristic methods.

2. Test/SyntheticExperiments.py

This contains the code for the simulation of synthetic data.

3. Test/RealDataExperiments.py

This contains the code to reproduce the deconvolution of the TCGA data.


