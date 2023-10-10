import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from fints_generation.models.utils import SlidingWindowDataset
from fints_generation.models.base import Generator
from fints_generation.models.tcn import TCNModule

import pandas as pd
import numpy as np


class TCNVAEModule(LightningModule):
    def __init__(
        self, latent_dim, hidden_dim, target_dim, kernel_size, num_layers, lr, dropout
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lr = lr

        self.enc = TCNModule(
            input_size=target_dim,
            kernel_size=kernel_size,
            num_filters=hidden_dim,
            num_layers=num_layers,
            dilation_base=2,
            weight_norm=False,
            target_size=hidden_dim,
            dropout=dropout,
        )
        self.dec = TCNModule(
            input_size=latent_dim,
            kernel_size=kernel_size,
            num_filters=hidden_dim,
            num_layers=num_layers,
            dilation_base=2,
            weight_norm=False,
            target_size=hidden_dim,
            dropout=dropout,
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)
        self.projection = nn.Linear(hidden_dim, target_dim)

    def neg_elbo(self, ts, rec_ts, sigma, mu):
        likelihood = -((rec_ts - ts)**2).sum((1,2))
        kld = -0.5 * (1 + (sigma**2 + 1e-10).log() - mu**2 - sigma**2).sum((1,2))
        elbo = likelihood - kld
        return -elbo.mean()

    def forward(self, target):
        # target shape: (batch size, seq len, num channels)
        h = self.enc(target) # (batch size, seq len, hidden_dim)
        mu, sigma = self.mu(h), torch.relu(self.sigma(h))
        return mu, sigma

    def reparametrization_trick(self, mu, sigma):
        z = mu + torch.randn_like(mu) * sigma
        return z

    def training_step(self, batch, batch_idx):
        target = batch
        mu, sigma = self.forward(target)
        z = self.reparametrization_trick(mu, sigma)
        h = self.dec(z)
        rec_target = self.projection(h)
        loss = self.neg_elbo(target, rec_target, sigma, mu)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=1)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'train_loss'
        }

    def sample(self, seq_len, n_samples):
        z = torch.randn(n_samples, seq_len, self.latent_dim, device=self.device)
        with torch.no_grad():
            h = self.dec(z)
            fake = self.projection(h).cpu()
        return fake


class TCNVAE(Generator):
    def __init__(
            self, 
            latent_dim=4,
            hidden_dim=16,
            kernel_size=2,
            num_layers=1,
            window_size=10, 
            batch_size=16, 
            num_epochs=1,
            lr=0.001,
            dropout=0,
            verbose=False,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.dropout = dropout
        self.verbose = verbose

    def fit(self, data):
        super().fit(data)
        self.model = TCNVAEModule(
            target_dim=len(self.columns), 
            hidden_dim=self.hidden_dim, 
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            lr=self.lr,
            dropout= self.dropout
        )
        self.dataset = SlidingWindowDataset(
            df=data,
            window_size=self.window_size,
            step_size=1,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.trainer = Trainer(
            enable_progress_bar=self.verbose,
            accelerator='auto',
            max_epochs=self.num_epochs,
            log_every_n_steps=np.ceil(len(self.dataloader) * 0.1),
            logger=TensorBoardLogger('.'),
        )
        self.trainer.fit(
            model=self.model, 
            train_dataloaders=self.dataloader,
        )

    def sample(self, index: pd.DatetimeIndex, n_samples: int) -> list:
        fakes = []
        for fake in self.model.sample(len(index), n_samples):
            fake_df = pd.DataFrame(fake, index=index, columns=self.columns)
            fakes.append(fake_df)
        return fakes
