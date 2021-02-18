import sys
import os
import torch
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from datetime import datetime
import logging, logging.handlers
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from MODEL.models import VAEModel
from MODEL.autoencoders import VAELinearAutoencoder
from MODEL.trainers import VAETrainer

parser = argparse.ArgumentParser()
parser.add_argument('--zdim', type=int, default=3, help='z dim')
parser.add_argument('--betareg', type=float, default=1.0, help='beta reg')
parser.add_argument('--lambdareg', type=float, default=1.0, help='lambdareg')
parser.add_argument('--epochsd', type=int, default=300, help='epochs for D')
parser.add_argument('--epochsy', type=int, default=400, help='epochs for Y')
parser.add_argument('--mlpsize', nargs='+', type=int, help='layers for MLP')
parser.add_argument('--maxiter', type=int, default=1000, help='max number of iterations for MLP')

args = parser.parse_args()

IN_DIM = 1
Z_DIM = args.zdim
X_DIM = 5
DATASET = "Simulation"
N_EPOCHS_D = args.epochsd
N_EPOCHS_Y = args.epochsy
TIME = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

def main():
    trainer_a_D = VAETrainer(VAEModel(VAELinearAutoencoder(IN_DIM, Z_DIM, X_DIM)), target="D", ab_index="a", lambda_reg=args.lambdareg, beta_reg=args.betareg, dataset=DATASET, time = TIME)
    trainer_a_D.train(N_EPOCHS_D)
    trainer_a_Y = VAETrainer(VAEModel(VAELinearAutoencoder(IN_DIM, Z_DIM, X_DIM)), target="Y", ab_index="a", lambda_reg=args.lambdareg, beta_reg=args.betareg, dataset=DATASET, time = TIME)
    trainer_a_Y.train(N_EPOCHS_Y)
    trainer_b_D = VAETrainer(VAEModel(VAELinearAutoencoder(IN_DIM, Z_DIM, X_DIM)), target="D", ab_index="b", lambda_reg=args.lambdareg, beta_reg=args.betareg, dataset=DATASET, time = TIME)
    trainer_b_D.train(N_EPOCHS_D)
    trainer_b_Y = VAETrainer(VAEModel(VAELinearAutoencoder(IN_DIM, Z_DIM, X_DIM)), target="Y", ab_index="b", lambda_reg=args.lambdareg, beta_reg=args.betareg, dataset=DATASET, time = TIME)
    trainer_b_Y.train(N_EPOCHS_Y)
    
    # predict D and Y from ML model but switch data sets
    trainer_a_D.predict() # get encodings for D on b
    trainer_a_Y.predict() # get encodings for D on a 
    trainer_b_D.predict() # get encodings for D on b
    trainer_b_Y.predict() # get encodings for D on a

    

    
    mean_b_D, targets_b_Y = trainer_a_D.pred_mean.cpu().numpy(), trainer_a_Y.pred_targets.cpu().numpy().ravel()
    mean_b_Y, targets_b_D = trainer_a_Y.pred_mean.cpu().numpy(), trainer_a_D.pred_targets.cpu().numpy().ravel()
    mean_a_D, targets_a_Y = trainer_b_D.pred_mean.cpu().numpy(), trainer_b_Y.pred_targets.cpu().numpy().ravel()
    mean_a_Y, targets_a_D = trainer_b_Y.pred_mean.cpu().numpy(), trainer_b_D.pred_targets.cpu().numpy().ravel()
    logging.info(mean_b_D, targets_b_Y)
    logging.info(mean_b_Y, targets_b_D)
    logging.info(mean_a_D, targets_a_Y)
    logging.info(mean_a_Y, targets_y_D)
    

    MLP_SIZE = tuple(args.mlpsize)
    MAXITER = args.maxiter
    logging.info(f"MLP size: {MLP_SIZE} | max iter: {MAXITER}")
    n_samples_a = len(mean_a_Y)
    n_samples_b = len(mean_b_Y)
    a_train, a_test = train_test_split(range(n_samples_a), test_size=0.5)
    b_train, b_test = train_test_split(range(n_samples_b), test_size=0.5)
    mlp_b_D = MLPRegressor(MLP_SIZE, max_iter=MAXITER).fit(mean_b_D[b_train], targets_b_D[b_train])
    mlp_b_Y = MLPRegressor(MLP_SIZE, max_iter=MAXITER).fit(mean_b_Y[b_train], targets_b_Y[b_train])
    mlp_a_D = MLPRegressor(MLP_SIZE, max_iter=MAXITER).fit(mean_a_D[a_train], targets_a_D[a_train])
    mlp_a_Y = MLPRegressor(MLP_SIZE, max_iter=MAXITER).fit(mean_a_Y[a_train], targets_a_Y[a_train])


    logging.info(f"train scores: {mlp_a_D.score(mean_a_D[a_train], targets_a_D[a_train])} | {mlp_b_D.score(mean_b_D[b_train], targets_b_D[b_train])}  | {mlp_a_Y.score(mean_a_Y[a_train], targets_a_Y[a_train])} | {mlp_b_Y.score(mean_b_Y[b_train], targets_b_Y[b_train])} test scores: {mlp_a_D.score(mean_a_D[a_test], targets_a_D[a_test])} | {mlp_b_D.score(mean_b_D[b_test], targets_b_D[b_test])} | {mlp_a_Y.score(mean_a_Y[a_test], targets_a_Y[a_test])} | {mlp_b_Y.score(mean_b_Y[b_test], targets_b_Y[b_test])}")


    mlp_b_D_pred = mlp_b_D.predict(mean_b_D[b_test])
    mlp_b_Y_pred = mlp_b_Y.predict(mean_b_Y[b_test])
    mlp_a_D_pred = mlp_a_D.predict(mean_a_D[a_test])
    mlp_a_Y_pred = mlp_a_Y.predict(mean_a_Y[a_test])


    reg_a = LinearRegression().fit(mlp_a_D_pred.reshape(-1, 1), mlp_a_Y_pred.reshape(-1, 1))
    reg_b = LinearRegression().fit(mlp_b_D_pred.reshape(-1, 1), mlp_b_Y_pred.reshape(-1, 1))

    logging.info(f"regression coefficients: A {reg_b.coef_} | B {reg_a.coef_}")

if __name__ == "__main__":
    main()