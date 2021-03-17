# Diffusion_u_net
The excessive deposition of misfolded proteins such as amyloid-$\beta$~(A$\beta$) protein is an aging event underlying several neurodegenerative diseases. Mounting evidence shows that the spreading of neuropathological burden has a strong association to the white matter tracts in the brain which can be measured using diffusion-weighted imaging and tractography technologies. Most of the previous studies analyze the dynamic progression of amyloid using cross-sectional data which is not robust to the heterogeneous A$\beta$ dynamics across the population. In this regard, we propose a graph neural network-based learning framework to capture the disease-related dynamics by tracking the spreading of amyloid across brain networks from the subject-specific longitudinal PET images. To learn from limited (2 – 3 timestamps) and noisy longitudinal data, we restrict the space of amyloid propagation patterns to a latent heat diffusion model which is constrained by the anatomical connectivity of the brain. Our experiments show that restricting the dynamics to be a heat diffusion mechanism helps to train a robust deep neural network for predicting future time points and classifying Alzheimer's disease brain.

# Installation
The dependencies can be install by --- pip install -r requirements.txt.

# Code organization
* reconstruction models: It contains the code for Diffusion u-net, Cluster u-net, Adaptive u-net, and the baseline methods for comparison.
* classification models: It contains the code for the classification models ($v_1$, \cdots, $v_5$)
* reconstruction trainer: Contains the trainer for reconstruction models
* classification trainer: Contains the trainer for classifiers
* layers: contains the code for GCNClassifier, Graph u-net etc.

## Training models
A simple example to train a reconstruction model
`python -m reconstruction_trainer.adaptive_u_net_trainer --dropout 0.1 --write_dir "adaptive_u_net" --modes 64 --batch_size 32 --lr 1e-5 --max_epoch 1000 --split 1500 --n 148 --train_r 0.7 --valid_r 0.15 --loss_w 1e2 1 0 1 --train True --ks 0.5 0.5 0.5 0.5 --indim 64`

