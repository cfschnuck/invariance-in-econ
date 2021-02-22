# RBJ Course Project 2020: Adjusting for Confounders with Invariant Representation Learning

## Introduction
This repository contains the code for reproducing the results of the course project in the 2020 Building a Robot Judge class at ETH Zurich. This project is mainly based on the paper ["Invariant Representations without Adversarial Training"](https://arxiv.org/abs/1805.09458). The project report can be found [here](RBJ_Project_Report.pdf).

## Installing dependencies
```bash
pip install -r requirements.txt
````

## Code structure
| Folder | Filename | Description |
|--------|----------|-------------|
| [MODEL](MODEL)   | [autoencoders.py](MODEL/autoencoders.py)| definition of variational autoencoder architecture|
| [MODEL](MODEL)   | [losses.py](MODEL/losses.py)| definitions of training losses|
| [MODEL](MODEL)   | [models.py](MODEL/models.py)| definition of variational autoencoder|
| [MODEL](MODEL)   | [trainers.py](MODEL/trainers.py)| model training utilities|
| [Preprocessing](Preprocessing)   | [helper_functions_preprocessing.py](Preprocessing/helper_functions_preprocessing.py)| helper functions used for preprocessing of data|
| [Preprocessing](Preprocessing)   | [nlsy.py](Preprocessing/nlsy.py)| script to preprocess and save NLSY dataset|
| [Preprocessing](Preprocessing)   | [simulations.py](Preprocessing/simulations.py)| script to preprocess and save simulated dataset|
| [Scripts](Scripts)   | [vae_train.py](Scripts/vae_train.py)| script to train VAE model and compare to DML and OLS|

## Reproduction of results
```bash
## Simulated dataset
python Scripts/vae_train.py --epochsd 150 --epochsy 250 --dataset Simulation
## NLSY dataset
python Scripts/vae_train.py --epochsd 150 --epochsy 200 --dataset NLSY
```


