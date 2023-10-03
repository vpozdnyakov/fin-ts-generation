import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from fints_generation.models.utils import SlidingWindowDataset
from fints_generation.models.base import Generator

import pandas as pd
import numpy as np


class CouplingLayerModule(LightningModule):
    def __init__(
        self, input_dim, hidden_dim, n_hidden, mask
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask = mask
        hlist = [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()] * n_hidden
        self.s = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            *hlist,
            nn.Linear(hidden_dim, input_dim), nn.Tanh(),
        )
        hlist = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * n_hidden
        self.t = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            *hlist,
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z0 = x * self.mask
        s = self.s(z0) * (1 - self.mask)
        t = self.t(z0) * (1 - self.mask)
        z1 = (x * torch.exp(s) + t) * (1 - self.mask)
        log_det_j = s.sum(dim=(2))
        return z0 + z1, log_det_j

    def inverse(self, z):
        x0 = z * self.mask
        s = self.s(x0) * (1 - self.mask)
        t = self.t(x0) * (1 - self.mask)
        x1 = (z - t) * torch.exp(-s) * (1 - self.mask)
        return x0 + x1


class RealNVPModule(LightningModule):
    def __init__(self, n_blocks, input_dim, hidden_dim, n_hidden, lr):
        super().__init__()
        self.lr = lr
        clist = []
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(input_dim), torch.eye(input_dim)
        )
        mask = (torch.arange(input_dim).float() % 2)
        for _ in range(n_blocks):
            clist.append(
                CouplingLayerModule(input_dim, hidden_dim, n_hidden, mask)
            )
            mask = 1 - mask
        self.flow = nn.ModuleList(clist)

    def forward(self, x):
        z = x
        log_det_j = torch.zeros(x.shape[:-1])
        for cl in self.flow:
            z, _log_det_j = cl.forward(z)
            log_det_j += _log_det_j

        return z, log_det_j

    def inverse(self, z):
        x = z
        for cl in self.flow[::-1]:
            x = cl.inverse(x)
        return x
    
    def training_step(self, batch, batch_idx):
        z, log_det_j = self.forward(batch)
        log_prob = self.prior.log_prob(z) + log_det_j
        loss = -log_prob.mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=3)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'train_loss'
        }
    
    def sample(self, length):
        z = self.prior.sample((1, length))
        gen_sample = self.inverse(z).cpu().detach().numpy()

        return gen_sample


class RealNVP(Generator):
    def __init__(
        self, 
        input_dim=4,
        hidden_dim=16,
        n_blocks=1,
        n_hidden=10,
        window_size=10,
        batch_size=16, 
        num_epochs=1,
        lr=0.001,
        verbose=False,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.verbose = verbose

    def fit(self, data):
        super().fit(data)
        self.model = RealNVPModule(
            n_blocks=self.n_blocks, 
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            n_hidden=self.n_hidden,
            lr=self.lr
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
        for fake in self.model.sample(len(index)):
            fake_df = pd.DataFrame(fake, index=index, columns=self.columns)
            fakes.append(fake_df)
            
        return fakes