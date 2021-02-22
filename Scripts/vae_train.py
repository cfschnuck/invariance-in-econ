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
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from MODEL.models import VAEModel
from MODEL.autoencoders import VAELinearAutoencoder
from MODEL.trainers import VAETrainer

parser = argparse.ArgumentParser()
parser.add_argument('--zdim', type=int, default=2, help='z dim')
parser.add_argument('--betareg', type=float, default=1, help='beta reg')
parser.add_argument('--lambdareg', type=float, default=1, help='lambdareg')
parser.add_argument('--epochsd', type=int, default=150, help='epochs for D')
parser.add_argument('--epochsy', type=int, default=200, help='epochs for Y')
# parser.add_argument('--mlpsize', nargs='+', type=int, help='layers for MLP')
parser.add_argument('--maxiter', type=int, default=2000, help='max number of iterations for MLP')
parser.add_argument('--dataset', type=str, default='NLSY', help='dataset to use')

args = parser.parse_args()

IN_DIM = 1
Z_DIM = args.zdim
DATASET = args.dataset
if DATASET == "Simulation":
    X_DIM = 5
elif DATASET == "NLSY":
    X_DIM = 9
else:
    raise NotImplementedError
N_EPOCHS_D = args.epochsd
N_EPOCHS_Y = args.epochsy
TIME = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

def main():
    trainer_a_D = VAETrainer(VAEModel(VAELinearAutoencoder(IN_DIM, Z_DIM, X_DIM, "D")), target="D", ab_index="a", lambda_reg=args.lambdareg, beta_reg=args.betareg, dataset=DATASET, time = TIME)
    trainer_a_D.train(N_EPOCHS_D)
    trainer_a_Y = VAETrainer(VAEModel(VAELinearAutoencoder(IN_DIM, Z_DIM, X_DIM, "Y")), target="Y", ab_index="a", lambda_reg=args.lambdareg, beta_reg=args.betareg, dataset=DATASET, time = TIME)
    trainer_a_Y.train(N_EPOCHS_Y)
    trainer_b_D = VAETrainer(VAEModel(VAELinearAutoencoder(IN_DIM, Z_DIM, X_DIM, "D")), target="D", ab_index="b", lambda_reg=args.lambdareg, beta_reg=args.betareg, dataset=DATASET, time = TIME)
    trainer_b_D.train(N_EPOCHS_D)
    trainer_b_Y = VAETrainer(VAEModel(VAELinearAutoencoder(IN_DIM, Z_DIM, X_DIM, "Y")), target="Y", ab_index="b", lambda_reg=args.lambdareg, beta_reg=args.betareg, dataset=DATASET, time = TIME)
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
    
    # setting parameters for prediction with MLP
    MLP_SIZE = (100, 50)
    MAXITER = args.maxiter
    logging.info(f"MLP size: {MLP_SIZE} | max iter: {MAXITER}")

    total_count = 0
    convergence_count = 0
    coef_list_ivae = []

    # predict Y and D from Ytilde and Dtilde until 10 times prediction was succesfull (MLP converged)
    while total_count <= 100 and convergence_count <= 10:
        logging.info(f"iteration: {total_count}")
        n_samples_a = len(mean_a_Y)
        n_samples_b = len(mean_b_Y)
        a_train, a_test = train_test_split(range(n_samples_a), test_size=0.5)
        b_train, b_test = train_test_split(range(n_samples_b), test_size=0.5)
        mlp_b_D = LinearRegression().fit(mean_b_D[b_train], targets_b_D[b_train])
        mlp_b_Y = MLPRegressor(MLP_SIZE, max_iter=MAXITER, early_stopping=True).fit(mean_b_Y[b_train], targets_b_Y[b_train])
        mlp_a_D = LinearRegression().fit(mean_a_D[a_train], targets_a_D[a_train])
        mlp_a_Y = MLPRegressor(MLP_SIZE, max_iter=MAXITER, early_stopping=True).fit(mean_a_Y[a_train], targets_a_Y[a_train])

        if DATASET == "Simulation":
            conv_limit = 0.7
        if DATASET == "NLSY":
            conv_limit = 0.5
        converged = mlp_a_D.score(mean_a_D[a_train], targets_a_D[a_train]) >= 0.7 and mlp_b_D.score(mean_b_D[b_train], targets_b_D[b_train]) >= 0.7 and mlp_a_Y.score(mean_a_Y[a_train], targets_a_Y[a_train]) >= conv_limit and mlp_b_Y.score(mean_b_Y[b_train], targets_b_Y[b_train]) >= conv_limit

        logging.info(f"MLP test scores: {mlp_a_D.score(mean_a_D[a_test], targets_a_D[a_test])} | {mlp_b_D.score(mean_b_D[b_test], targets_b_D[b_test])} | {mlp_a_Y.score(mean_a_Y[a_test], targets_a_Y[a_test])} | {mlp_b_Y.score(mean_b_Y[b_test], targets_b_Y[b_test])}")

        if converged:
            mlp_b_D_pred = mlp_b_D.predict(mean_b_D[b_test])
            mlp_b_Y_pred = mlp_b_Y.predict(mean_b_Y[b_test])
            mlp_a_D_pred = mlp_a_D.predict(mean_a_D[a_test])
            mlp_a_Y_pred = mlp_a_Y.predict(mean_a_Y[a_test])

            reg_a = LinearRegression().fit(mlp_a_D_pred.reshape(-1, 1), mlp_a_Y_pred.reshape(-1, 1))
            reg_b = LinearRegression().fit(mlp_b_D_pred.reshape(-1, 1), mlp_b_Y_pred.reshape(-1, 1))

            logging.info(f"class regression coefficients: A {reg_b.coef_} | B {reg_a.coef_}")
            logging.info(f"class regression score: A {reg_b.score(mlp_a_D_pred.reshape(-1, 1), mlp_a_Y_pred.reshape(-1, 1))} | B {reg_a.score(mlp_b_D_pred.reshape(-1, 1), mlp_b_Y_pred.reshape(-1, 1))}")

            coef_list_ivae.append((reg_b.coef_ + reg_a.coef_) / 2)

            convergence_count += 1
        
        # log coefficients for IVAE
        if convergence_count > 10 or total_count > 100:
            logging.info(f"final coefficients IVAE:")
            for i in range(len(coef_list_ivae)):
                logging.info(f"{coef_list_ivae[i]}")

        total_count += 1
        if total_count == 100:
            logging.info(f"WARNING: not converged after 100 iterations")

    # biased regression for comparison
    reg_a_biased = LinearRegression().fit(targets_a_D.reshape(-1, 1), targets_a_Y.reshape(-1, 1))
    reg_b_biased = LinearRegression().fit(targets_b_D.reshape(-1, 1), targets_b_Y.reshape(-1, 1))
    logging.info(f"regression coefficients: A {reg_b_biased.coef_} | B {reg_a_biased.coef_}")
    logging.info(f"regression score: A {reg_b_biased.score(targets_a_D.reshape(-1, 1), targets_a_Y.reshape(-1, 1))} | B {reg_a_biased.score(targets_b_D.reshape(-1, 1), targets_b_Y.reshape(-1, 1))}")
    coef_ols = (reg_b_biased.coef_ + reg_a_biased.coef_) / 2
    logging.info(f"Coefficient from OLS: {coef_ols}")

    # Double ML approach
    coef_list_dml = []
    PATH_DATASET = "/Users/carina/Documents/Uni/ETH/CodeProjects/invariance-in-econ/Data/Preprocessed/" + DATASET + "/"
    MLP_SIZE = (20,10)
    for i in range(10):
        Y_a_train = torch.load(PATH_DATASET + "Y" + "_" + "a" + "_" + "train").cpu().numpy()
        D_a_train = torch.load(PATH_DATASET + "D" + "_" + "a" + "_" + "train").cpu().numpy()
        X_a_train = torch.load(PATH_DATASET + "X" + "_" + "a" + "_" + "train").cpu().numpy()
        Y_b_train = torch.load(PATH_DATASET + "Y" + "_" + "b" + "_" + "train").cpu().numpy()
        D_b_train = torch.load(PATH_DATASET + "D" + "_" + "b" + "_" + "train").cpu().numpy()
        X_b_train = torch.load(PATH_DATASET + "X" + "_" + "b" + "_" + "train").cpu().numpy()

        # fit on sample A
        mlp_d_a = MLPClassifier(MLP_SIZE, max_iter=MAXITER, early_stopping=True).fit(X_a_train, D_a_train.ravel())
        mlp_y_a = MLPRegressor(MLP_SIZE, max_iter=MAXITER, early_stopping=True).fit(X_a_train, Y_a_train.ravel())
        # fit on sample B
        mlp_d_b = MLPClassifier(MLP_SIZE, max_iter=MAXITER, early_stopping=True).fit(X_b_train, D_b_train.ravel())
        mlp_y_b = MLPRegressor(MLP_SIZE, max_iter=MAXITER, early_stopping=True).fit(X_b_train, Y_b_train.ravel())

        # predict on sample B
        D_b_pred = mlp_d_a.predict_proba(X_b_train)[:,1]
        Y_b_pred = mlp_y_a.predict(X_b_train)
        # predict on sample A
        D_a_pred = mlp_d_b.predict_proba(X_a_train)[:,1]
        Y_a_pred = mlp_y_b.predict(X_a_train)
        # form residuals
        D_a_res = D_a_train - D_a_pred.reshape(-1,1)
        Y_a_res = Y_a_train - Y_a_pred.reshape(-1,1)
        D_b_res = D_b_train - D_b_pred.reshape(-1,1)
        Y_b_res = Y_b_train - Y_b_pred.reshape(-1,1)

        reg_a = LinearRegression().fit(D_a_res, Y_a_res)
        reg_b = LinearRegression().fit(D_b_res, Y_b_res)

        reg_dml = (reg_a.coef_ + reg_b.coef_) / 2
        coef_list_dml.append(reg_dml)

    # plot coefficients of OLS, DML and VAE
    plt.figure()
    plt.scatter(coef_list_ivae, np.zeros_like(coef_list_ivae), c="blue", s=5, alpha=1, label = "VAE")
    plt.scatter(coef_list_dml, np.zeros_like(coef_list_dml), c="green", s=5, alpha=1, label = "DML")
    plt.scatter(coef_ols, 0, c="red", s=5, alpha=1, label = "OLS")
    plt.legend()
    plt.title("Estimated Coefficients for VAE, DML and OLS")
    plt.tight_layout()
    plt.savefig(trainer_b_Y.model_path_main + "coef_plot")


if __name__ == "__main__":
    main()