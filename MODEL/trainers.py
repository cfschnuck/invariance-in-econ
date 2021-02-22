import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchnet.meter import AverageValueMeter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import abstractmethod
import shutil
import os
from os.path import dirname, realpath
import logging, logging.handlers
from datetime import datetime
from .losses import *






class VAETrainer():
    """Trainer for Vae Model

    Args:
        model: model architecture
        target: indicator whether data is treatment variable D or outcome variable Y
        ab_index: indicator whether dataset A or B is used for training
        lambda_reg: lambda perameter controlling importance of mutual information criterion in loss
        beta_reg: beta perameter controlling importance of KL divergence to prior in loss
        batch_size: batch size
        lr: learning rate
        dataset: dataset. Defaults to Simulation.
        time: time to be used for model path for saving files
    """

    def __init__(self, model, target, ab_index, lambda_reg = 1e-1, beta_reg = 1e-1
    , batch_size=264, lr=1e-4, dataset="Simulation", time=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.target = target
        self.ab_index = ab_index
        # loss weight parameters
        self.lambda_reg = lambda_reg
        self.beta_reg = beta_reg
        # losses
        self.reconstruction_loss = F.mse_loss
        self.prior_loss = prior_loss_f
        self.kl_qzx_qz_loss = kl_qzx_qz_loss_f
        # training parameters
        self.lr = lr
        # dataset and model saving configurations
        self.dataset = dataset
        self.path_dataset = "Data/Preprocessed/" + self.dataset + "/"
        self.load_data()
        self.optimizer = optim.Adam(self.model.parameters(), lr= self.lr)
        #logging info
        self.time = time
        self.model_path = dirname(dirname(realpath(__file__))) + f"/Pretrained/{self.dataset}/{self.model.__class__.__name__}/{self.time}/{self.target}{self.ab_index}/" 
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_path_main = dirname(dirname(realpath(__file__))) + f"/Pretrained/{self.dataset}/{self.model.__class__.__name__}/{self.time}/"
        logging.basicConfig(filename=dirname(dirname(realpath(__file__))) + f"/Pretrained/{self.dataset}/{self.model.__class__.__name__}/{self.time}/train.log", level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    def create_dataset(self, path_dataset, target, ab_index, train):
        X = torch.load(path_dataset + "X" + "_" + ab_index + "_" + train)
        target = torch.load(path_dataset + target + "_" + ab_index + "_" + train)
        return TensorDataset(X, target)

    def load_data(self):
        self.train_dataset = self.create_dataset(self.path_dataset, self.target, self.ab_index, "train")
        self.test_dataset = self.create_dataset(self.path_dataset, self.target, self.ab_index, "test")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def save_checkpoint(self, state, is_best):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        filename = self.model_path + 'checkpoint.pth.tar.gz'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.model_path + 'model_best.pth.tar.gz')

    def load_checkpoint(self, path):
        try:
            self.state_checkpoint = torch.load(path + 'checkpoint.pth.tar.gz', map_location=self.device)
            self.state_best_model = torch.load(path + 'model_best.pth.tar.gz', map_location=self.device)
        except FileNotFoundError:
            logging.warning("No pretrained model found. Training continues without pretrained weights.")
        else:
            self.model.load_state_dict(self.state_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.state_checkpoint['optimizer_state_dict'])
            self.best_model = copy.deepcopy(self.model)
            self.best_model.load_state_dict(self.state_best_model['model_state_dict'])
            self.model_params = self.state_best_model['params']
    
    @property
    def model_params(self):
        return {"lr" : self.lr, "batch_size": self.batch_size, "lambda" : self.lambda_reg, "beta" : self.beta_reg}

    def _log_start(self):
        logging.info(f"Model params: | lambda: {self.lambda_reg} | beta: {self.beta_reg} | z dim: {self.model.autoencoder.z_dim}")

    def _init_eval(self):
        self.eval_dict = {"train reconstruction loss" : [], "train prior loss" : [], "train paired gkl loss" : [], "test reconstruction loss" : []}

    def _update_eval(self):
        self.eval_dict["train reconstruction loss"].append(self.reconstruction_loss_meter.mean * (1 + self.lambda_reg))
        self.eval_dict["train prior loss"].append(self.prior_loss_meter.mean * self.beta_reg)
        self.eval_dict["train paired gkl loss"].append(self.kl_qzx_qz_loss_meter.mean * self.lambda_reg)
        self.eval_dict["test reconstruction loss"].append(self.reconstruction_loss_meter_val.mean)

    def _save_eval(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        filename = self.model_path + 'eval_dict'
        torch.save(self.eval_dict, filename)
    
    def train(self, n_epochs):
        best_score = 0
        self._log_start()
        self._init_eval()
        for epoch in tqdm(range(0, n_epochs), desc="Epoch: "):
            self.reconstruction_loss_meter = AverageValueMeter()
            self.prior_loss_meter = AverageValueMeter()
            self.kl_qzx_qz_loss_meter = AverageValueMeter()
            self.total_loss_meter = AverageValueMeter()
            self.model.train()
            for batch_id, (x, targets) in enumerate(self.train_dataloader):
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                self.targets.requires_grad_(True)
                self.model.zero_grad()
                z_mean, z_log_var, targets_reconstructed = self.model(self.targets, self.x)
                reconstruction_loss = self.reconstruction_loss(targets_reconstructed, self.targets)
                prior_loss = self.prior_loss(z_mean, z_log_var)
                kl_qzx_qz_loss = self.kl_qzx_qz_loss(z_mean, z_log_var)
                total_loss = (1 + self.lambda_reg) * reconstruction_loss + self.beta_reg * prior_loss + self.lambda_reg * kl_qzx_qz_loss # TODO
                total_loss.backward()
                self.optimizer.step()
                self.reconstruction_loss_meter.add(reconstruction_loss.item())
                self.prior_loss_meter.add(prior_loss.item())
                self.kl_qzx_qz_loss_meter.add(kl_qzx_qz_loss.item())
                self.total_loss_meter.add(total_loss.item())
            self._log_epoch_full(epoch)
            self.validate()
            self._update_eval()
            is_best = self.reconstruction_loss_meter_val.mean > best_score
            best_score = max(self.reconstruction_loss_meter_val.mean, best_score)
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'train acc': self.reconstruction_loss_meter.mean,
                    'val acc': self.reconstruction_loss_meter_val.mean,
                    'optimizer_state_dict' : self.optimizer.state_dict(), 
                    'total_loss' : total_loss,
                    'params' : self.model_params
                    }, is_best)
        self.save_loss_img()

    def validate(self):
        self.model.eval()
        self.reconstruction_loss_meter_val = AverageValueMeter()
        with torch.no_grad():
            for x, targets in self.test_dataloader:
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                z_mean, z_log_var, targets_reconstructed = self.model(self.targets, self.x)
                self.reconstruction_loss_meter_val.add(self.reconstruction_loss(targets_reconstructed, self.targets).detach().cpu())
            logging.info(f"Acc: {self.reconstruction_loss_meter_val.mean}")

    def _log_epoch_full(self, epoch):
        logging.info(f"Epoch: {epoch} | Total Loss: {self.total_loss_meter.mean} | Reconstruction Loss: {self.reconstruction_loss_meter.mean} | Prior Loss: {self.prior_loss_meter.mean} | Paired gkl loss: {self.kl_qzx_qz_loss_meter.mean}")

    def predict(self):
        """Use learned VAE to encode latent variables on the left-out dataset
        """
        self.pred_target = self.target
        self.pred_ab_index = "b" if self.ab_index == "a" else "a"
        self.pred_X = torch.load(self.path_dataset + "X" + "_" + self.pred_ab_index + "_" + "train")
        self.pred_X = torch.zeros_like(self.pred_X)
        self.pred_targets = torch.load(self.path_dataset + self.pred_target + "_" + self.pred_ab_index + "_" + "train")
        self.pred_X, self.pred_targets = Variable(self.pred_X).to(self.device), Variable(self.pred_targets).to(self.device)
        # self.load_pred_data()
        self.model.eval()
        with torch.no_grad():
            self.pred_mean, _, _ = self.model(self.pred_targets, self.pred_X)
        self.save_latent_var_img()

    def save_loss_img(self):
        """ First create image for parts of training loss over epochs. Second create image for train and test loss reconstruction loss over epochs.
        """
        x = range(len(self.eval_dict["train reconstruction loss"]))
        fig, ax = plt.subplots()
        ax.plot(x, self.eval_dict["train reconstruction loss"], label="train reconstruction mse")
        ax.plot(x, self.eval_dict["test reconstruction loss"], label="test reconstruction mse")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.title("Reconstruction Loss")
        plt.tight_layout()
        plt.savefig(self.model_path + 'reconstruction_loss.png')

        x = range(len(self.eval_dict["train reconstruction loss"]))
        fig, ax = plt.subplots()
        ax.plot(x, self.eval_dict["train prior loss"], label="train KL divergence to prior")
        ax.plot(x, self.eval_dict["train paired gkl loss"], label="train KL divergence to marginal posterior")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.title("Variational Loss")
        plt.tight_layout()
        plt.savefig(self.model_path + 'train_total_loss.png')

    def save_latent_var_img(self):
        """Create plot to visualize latent variables in 2D (possibly via PCA if z_dim > 2)
        """
        if self.model.autoencoder.z_dim >= 2: # check if plot can be done in 2D
            if self.target == "D":
                pca = PCA(n_components=2)
                # randomly plot 1000 test samples
                _, ind = train_test_split(range(len(self.pred_mean)), test_size=1000)
                latents = self.pred_mean[ind].cpu().numpy()
                targets = self.pred_targets[ind].cpu().numpy().squeeze()
                X = pca.fit(latents).transform(latents)
                plt.figure()
                for i in zip([0, 1]):
                    plt.scatter(X[targets == i, 0], X[targets == i, 1], s=2, alpha=0.6, label=str(i))
                plt.title("PCA for latent variable decomposition of " + self.target)
                plt.tight_layout()
                plt.savefig(self.model_path + 'latents_PCA.png')
                if self.model.autoencoder.z_dim == 2:
                    plt.figure()
                    for i in zip([0, 1]):
                        plt.scatter(latents[targets == i, 0], latents[targets == i, 1], s=2, alpha=0.6, label=str(i))
                    plt.title("Latent Representation of " + self.target)
                    plt.tight_layout()
                    plt.savefig(self.model_path + 'latents.png')
            if self.target == "Y":
                pca = PCA(n_components=2)
                # randomly plot 1000 test samples
                _, ind = train_test_split(range(len(self.pred_mean)), test_size=1000)
                latents = self.pred_mean[ind].cpu().numpy()
                targets = self.pred_targets[ind].cpu().numpy().squeeze()
                X = pca.fit(latents).transform(latents)
                plt.figure()
                plt.scatter(X[:, 0], X[:, 1], s=2, alpha=0.6)
                plt.title("Latent variable decomposition of " + self.target)
                plt.tight_layout()
                plt.savefig(self.model_path + 'latents_PCA.png')
                if self.model.autoencoder.z_dim == 2:
                    plt.figure()
                    plt.scatter(latents[:, 0], latents[:, 1], s=2, alpha=0.6)
                    plt.title("Latent Representation of " + self.target)
                    plt.tight_layout()
                    plt.savefig(self.model_path + 'latents.png')




                
