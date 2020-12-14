import sys
import os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import numpy as np
from sklearn.linear_model import LinearRegression

from MODEL.models import InvarSennM1, InvarSennM2
from MODEL.autoencoders import LinearAutoencoder
from MODEL.predictors import InvarPredictor
from MODEL.trainers import Trainer
from MODEL.disentanglers import Disentangler

N_E1 = 2
N_E2 = 5
N_EPOCHS = 10
IN_DIM = 1
OUT_DIM = 19

def main():
    # autoencoder = LinearAutoencoder(19, N_E1, N_E2)
    # predictor = InvarPredictor(N_E1)
    # disentangler1 = Disentangler(N_E1, N_E2)
    # disentangler2 = Disentangler(N_E2, N_E1)
    #m1 = InvarSennM1(autoencoder, predictor)
    #m2 = InvarSennM2(disentangler1, disentangler2)
    # train ML model and predict
    trainer_a_D = Trainer(InvarSennM1(LinearAutoencoder(IN_DIM, N_E1, N_E2, OUT_DIM), InvarPredictor(N_E1)), InvarSennM2(Disentangler(N_E1, N_E2), Disentangler(N_E2, N_E1)), target="D", ab_index="a")
    trainer_a_D.train(N_EPOCHS)
    trainer_a_Y = Trainer(InvarSennM1(LinearAutoencoder(IN_DIM, N_E1, N_E2, OUT_DIM), InvarPredictor(N_E1)), InvarSennM2(Disentangler(N_E1, N_E2), Disentangler(N_E2, N_E1)), target="Y", ab_index="a")
    trainer_a_Y.train(N_EPOCHS)
    trainer_b_D = Trainer(InvarSennM1(LinearAutoencoder(IN_DIM, N_E1, N_E2, OUT_DIM), InvarPredictor(N_E1)), InvarSennM2(Disentangler(N_E1, N_E2), Disentangler(N_E2, N_E1)), target="D", ab_index="b")
    trainer_b_D.train(N_EPOCHS)
    trainer_b_Y = Trainer(InvarSennM1(LinearAutoencoder(IN_DIM, N_E1, N_E2, OUT_DIM), InvarPredictor(N_E1)), InvarSennM2(Disentangler(N_E1, N_E2), Disentangler(N_E2, N_E1)), target="Y", ab_index="b")
    trainer_b_Y.train(N_EPOCHS)
    # predict D and Y from ML model
    trainer_a_D.predict()
    trainer_a_Y.predict()
    trainer_b_D.predict()
    trainer_b_Y.predict()
    #regress Y pred on D had and average out
    reg_b = LinearRegression().fit(trainer_a_D.pred.cpu().numpy(), trainer_a_Y.pred.cpu().numpy()) # regresses Y on D for data set B
    reg_a = LinearRegression().fit(trainer_b_D.pred.cpu().numpy(), trainer_b_Y.pred.cpu().numpy()) # regresses Y on D for data set A

    print(reg_b.coef_, reg_a.coef_)

    # train post hoc neural net
    # predict x from e1

    # predict target form e1

if __name__ == "__main__":
    main()