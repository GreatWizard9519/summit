## Overview 

This repository contains the code of paper: Structure Unlearning of Graph Data Using Multi-Objective Optimization. (SUMMIT)
The backbone of the code is based on Harvard Zitnik Lab's graph unlearning research project: https://zitniklab.hms.harvard.edu/projects/GNNDelete/.

## Environment

python -- 3.8.11
numpy -- 1.21.2
torch -- 1.9.1
torch_geometric -- 2.0.3
cuda -- 11.1


## Execution

1. Run "prepare_dataset.py" to download and preprocess the datasets.
2. Run ''training_main.py" to train the GNN model.
3. Run ''unlearning_main.py" to unlearn the GNN model.



