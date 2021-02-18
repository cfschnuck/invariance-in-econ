import sys
import os
import torch
import statistics
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from torch.nn.functional import mse_loss

from MODEL.models import InvarSennM1, InvarSennM2
from MODEL.autoencoders import LinearAutoencoder
from MODEL.predictors import InvarPredictor
from MODEL.trainers import AdvTrainer
from MODEL.disentanglers import Disentangler

N_E1 = 3
N_E2 = 5
N_EPOCHS = 400
IN_DIM = 1
OUT_DIM = 5
DATASET = "NLSY"

def main():
    # autoencoder = LinearAutoencoder(19, N_E1, N_E2)
    # predictor = InvarPredictor(N_E1)
    # disentangler1 = Disentangler(N_E1, N_E2)
    # disentangler2 = Disentangler(N_E2, N_E1)
    #m1 = InvarSennM1(autoencoder, predictor)
    #m2 = InvarSennM2(disentangler1, disentangler2)
    # train ML model and predict
    trainer_a_D = AdvTrainer(InvarSennM1(LinearAutoencoder(IN_DIM, N_E1, N_E2, OUT_DIM), InvarPredictor(N_E1)), InvarSennM2(Disentangler(N_E1, N_E2), Disentangler(N_E2, N_E1)), target="D", ab_index="a", dataset="Simulation")
    trainer_a_D.train(int(N_EPOCHS / 2))
    trainer_a_Y = AdvTrainer(InvarSennM1(LinearAutoencoder(IN_DIM, N_E1, N_E2, OUT_DIM), InvarPredictor(N_E1)), InvarSennM2(Disentangler(N_E1, N_E2), Disentangler(N_E2, N_E1)), target="Y", ab_index="a", dataset="Simulation")
    trainer_a_Y.train(N_EPOCHS)
    trainer_b_D = AdvTrainer(InvarSennM1(LinearAutoencoder(IN_DIM, N_E1, N_E2, OUT_DIM), InvarPredictor(N_E1)), InvarSennM2(Disentangler(N_E1, N_E2), Disentangler(N_E2, N_E1)), target="D", ab_index="b", dataset="Simulation")
    trainer_b_D.train(int(N_EPOCHS / 2))
    trainer_b_Y = AdvTrainer(InvarSennM1(LinearAutoencoder(IN_DIM, N_E1, N_E2, OUT_DIM), InvarPredictor(N_E1)), InvarSennM2(Disentangler(N_E1, N_E2), Disentangler(N_E2, N_E1)), target="Y", ab_index="b", dataset="Simulation")
    trainer_b_Y.train(N_EPOCHS)


    # predict D and Y from ML model but switch data sets
    trainer_a_D.predict() # predicts D on b
    trainer_a_Y.predict() # predicts Y on b 
    trainer_b_D.predict() # predicts D on a
    trainer_b_Y.predict() # predicts Y on a
    #regress Y pred on D had and average out
    reg_b = LinearRegression().fit(trainer_a_D.pred.cpu().numpy(), trainer_a_Y.pred.cpu().numpy()) # regresses Y on D for data set B
    reg_a = LinearRegression().fit(trainer_b_D.pred.cpu().numpy(), trainer_b_Y.pred.cpu().numpy()) # regresses Y on D for data set A

    print(reg_b.coef_, reg_a.coef_)

    # predict target form e1, calculate score for predicition and target
    path_dataset = trainer_a_D.path_dataset
    Y_a_train = torch.load(path_dataset + "Y" + "_" + "a" + "_" + "train")
    Y_b_train = torch.load(path_dataset + "Y" + "_" + "b" + "_" + "train")
    Y_train = torch.cat((Y_a_train, Y_b_train), 0)
    Y_std = Y_train.std()
    Y_pred = torch.cat((trainer_b_Y.pred, trainer_a_Y.pred), 0)
    Y_smse = mse_loss(Y_train, Y_pred).sqrt()
    print("sd Y:", Y_std)
    print("smse Y:", Y_smse)

    D_a_train = torch.load(path_dataset + "D" + "_" + "a" + "_" + "train")
    D_b_train = torch.load(path_dataset + "D" + "_" + "b" + "_" + "train")
    D_train = torch.cat((D_a_train, D_b_train), 0)
    D_std = D_train.std()
    D_pred = torch.cat((trainer_b_D.pred, trainer_a_D.pred), 0)
    D_smse = mse_loss(D_train, D_pred).sqrt()
    print("sd D:", D_std)
    print("smse D:", D_smse)

    mlp12_D_b_score = []
    mlp12_D_a_score = []
    mlp12_Y_b_score = []
    mlp12_Y_a_score = []
    mlp21_D_b_score = []
    mlp21_D_a_score = []
    mlp21_Y_b_score = []
    mlp21_Y_a_score = []
    mlp12_D_b_rand_score = []
    mlp12_D_a_rand_score = []
    mlp12_Y_b_rand_score = []
    mlp12_Y_a_rand_score = []
    mlp21_D_b_rand_score = []
    mlp21_D_a_rand_score = []
    mlp21_Y_b_rand_score = []
    mlp21_Y_a_rand_score = []

    MLP_SIZE = (20, 10)
    n_samples_a = D_a_train.size()[0]
    n_samples_b = D_b_train.size()[0]
    for i in range(10):
        a_train, a_test = train_test_split(range(n_samples_a), test_size=0.8)
        b_train, b_test = train_test_split(range(n_samples_b), test_size=0.8)
        # checking for disentanglement
        mlp12_D_b = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_a_D.pred_e1[b_train,:], trainer_a_D.pred_e2[b_train])
        mlp12_D_a = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_b_D.pred_e1[a_train,:], trainer_b_D.pred_e2[a_train])
        mlp12_Y_b = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_a_Y.pred_e1[b_train,:], trainer_a_Y.pred_e2[b_train])
        mlp12_Y_a = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_b_Y.pred_e1[a_train,:], trainer_b_Y.pred_e2[a_train])
        mlp21_D_b = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_a_D.pred_e2[b_train,:], trainer_a_D.pred_e1[b_train])
        mlp21_D_a = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_b_D.pred_e2[a_train,:], trainer_b_D.pred_e1[a_train])
        mlp21_Y_b = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_a_Y.pred_e2[b_train,:], trainer_a_Y.pred_e1[b_train])
        mlp21_Y_a = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_b_Y.pred_e2[a_train,:], trainer_b_Y.pred_e1[a_train])

        
        mlp12_D_b_score.append(mlp12_D_b.score(trainer_a_D.pred_e1[b_test,:], trainer_a_D.pred_e2[b_test,:]))
        mlp12_D_a_score.append(mlp12_D_a.score(trainer_b_D.pred_e1[a_test,:], trainer_b_D.pred_e2[a_test,:]))
        mlp12_Y_b_score.append(mlp12_Y_b.score(trainer_a_Y.pred_e1[b_test,:], trainer_a_Y.pred_e2[b_test,:]))
        mlp12_Y_a_score.append(mlp12_Y_a.score(trainer_b_Y.pred_e1[a_test,:], trainer_b_Y.pred_e2[a_test,:]))
        mlp21_D_b_score.append(mlp21_D_b.score(trainer_a_D.pred_e2[b_test,:], trainer_a_D.pred_e1[b_test,:]))
        mlp21_D_a_score.append(mlp21_D_a.score(trainer_b_D.pred_e2[a_test,:], trainer_b_D.pred_e1[a_test,:]))
        mlp21_Y_b_score.append(mlp21_Y_b.score(trainer_a_Y.pred_e2[b_test,:], trainer_a_Y.pred_e1[b_test,:]))
        mlp21_Y_a_score.append(mlp21_Y_a.score(trainer_b_Y.pred_e2[a_test,:], trainer_b_Y.pred_e1[a_test,:]))

        # random permutation
        rand_ind_a = torch.randperm(n_samples_a)
        rand_ind_b = torch.randperm(n_samples_b)
        rand_a_train = rand_ind_a[a_train]
        rand_a_test = rand_ind_a[a_test]
        rand_b_train = rand_ind_b[b_train]
        rand_b_test = rand_ind_b[b_test]
        mlp12_D_b = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_b_D.pred_e1[a_train,:], trainer_b_D.pred_e2[rand_a_train,:])
        mlp12_D_a = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_a_D.pred_e1[b_train,:], trainer_a_D.pred_e2[rand_b_train,:])
        mlp12_Y_a = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_b_Y.pred_e1[a_train,:], trainer_b_Y.pred_e2[rand_a_train,:])
        mlp12_Y_b = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_a_Y.pred_e1[b_train,:], trainer_a_Y.pred_e2[rand_b_train,:])
        mlp21_D_a = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_b_D.pred_e2[a_train,:], trainer_b_D.pred_e1[rand_a_train,:])
        mlp21_D_b = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_a_D.pred_e2[b_train,:], trainer_a_D.pred_e1[rand_b_train,:])
        mlp21_Y_a = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_b_Y.pred_e2[a_train,:], trainer_b_Y.pred_e1[rand_a_train,:])
        mlp21_Y_b = MLPRegressor(MLP_SIZE, max_iter=300).fit(trainer_a_Y.pred_e2[b_train,:], trainer_a_Y.pred_e1[rand_b_train,:])

        
        mlp12_D_b_rand_score.append(mlp12_D_b.score(trainer_b_D.pred_e1[a_test,:], trainer_b_D.pred_e2[rand_a_test,:]))
        mlp12_D_a_rand_score.append(mlp12_D_a.score(trainer_a_D.pred_e1[b_test,:], trainer_a_D.pred_e2[rand_b_test,:]))
        mlp12_Y_b_rand_score.append(mlp12_Y_a.score(trainer_b_Y.pred_e1[a_test,:], trainer_b_Y.pred_e2[rand_a_test,:]))
        mlp12_Y_a_rand_score.append(mlp12_Y_b.score(trainer_a_Y.pred_e1[b_test,:], trainer_a_Y.pred_e2[rand_b_test,:]))
        mlp21_D_b_rand_score.append(mlp21_D_a.score(trainer_b_D.pred_e2[a_test,:], trainer_b_D.pred_e1[rand_a_test,:]))
        mlp21_D_a_rand_score.append(mlp21_D_b.score(trainer_a_D.pred_e2[b_test,:], trainer_a_D.pred_e1[rand_b_test,:]))
        mlp21_Y_b_rand_score.append(mlp21_Y_a.score(trainer_b_Y.pred_e2[a_test,:], trainer_b_Y.pred_e1[rand_a_test,:]))
        mlp21_Y_a_rand_score.append(mlp21_Y_b.score(trainer_a_Y.pred_e2[b_test,:], trainer_a_Y.pred_e1[rand_b_test,:]))

    print("scores for e1 / e2 predictions")
    print(statistics.mean(mlp12_D_b_score))
    print(statistics.mean(mlp12_D_a_score))
    print(statistics.mean(mlp12_Y_b_score))
    print(statistics.mean(mlp12_Y_a_score))
    print(statistics.mean(mlp21_D_b_score))
    print(statistics.mean(mlp21_D_a_score))
    print(statistics.mean(mlp21_Y_b_score))
    print(statistics.mean(mlp21_Y_a_score))
    print("scores for random prediction")
    print(statistics.mean(mlp12_D_b_rand_score))
    print(statistics.mean(mlp12_D_a_rand_score))
    print(statistics.mean(mlp12_Y_b_rand_score))
    print(statistics.mean(mlp12_Y_a_rand_score))
    print(statistics.mean(mlp21_D_b_rand_score))
    print(statistics.mean(mlp21_D_a_rand_score))
    print(statistics.mean(mlp21_Y_b_rand_score))
    print(statistics.mean(mlp21_Y_a_rand_score))

if __name__ == "__main__":
    main()