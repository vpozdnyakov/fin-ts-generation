import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from fints_generation.models.utils import SlidingWindowDataset
from fints_generation.models.base import Generator
from fints_generation.models.tcn import TCNModule

import pandas as pd
import numpy as np


class _TCNGenerator(nn.Module):
    def __init__(self, latent_size, kernel_size, hidden_size, target_size, num_layers, dropout, weight_norm):
        super().__init__()
        self.latent_size = latent_size
        self.tcn = TCNModule(
            input_size=latent_size,
            kernel_size=kernel_size,
            num_filters=hidden_size,
            num_layers=num_layers,
            dilation_base=2,
            weight_norm=weight_norm,
            target_size=hidden_size,
            dropout=dropout,
        )
        self.projection = nn.Linear(hidden_size, target_size)

    def forward(self, z):
        output = self.tcn(z)
        fake = self.projection(output)
        return fake


class _TCNDiscriminator(nn.Module):
    def __init__(self, target_size, kernel_size, hidden_size, num_layers, dropout, weight_norm):
        super().__init__()
        self.tcn = TCNModule(
            input_size=target_size,
            kernel_size=kernel_size,
            num_filters=hidden_size,
            num_layers=num_layers,
            dilation_base=2,
            weight_norm=weight_norm,
            target_size=hidden_size,
            dropout=dropout,
        )
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, target):
        output = self.tcn(target)
        logits = self.projection(output)
        return logits, output
        

class TCNGANModule(LightningModule):
    def __init__(
        self, 
        input_size=3, 
        hidden_size=80, 
        latent_size=3, 
        num_layers=6, 
        dropout=0, 
        generator_lr=1e-5, 
        discriminator_lr=1e-5, 
        num_disc_steps=1,
        clip_discriminator_weights=0,
        clip_generator_weights=0,
        weight_norm=False
    ):
        super().__init__()
        self.automatic_optimization = False
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.num_disc_steps = num_disc_steps
        self.dropout = dropout
        self.clip_discriminator_weights = clip_discriminator_weights
        self.clip_generator_weights = clip_generator_weights
        self.generator = _TCNGenerator(
            latent_size=latent_size,
            target_size=input_size, 
            kernel_size=2, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout,
            weight_norm=weight_norm
        )
        self.discriminator = _TCNDiscriminator(
            target_size=input_size, 
            kernel_size=2, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout,
            weight_norm=weight_norm
        )

    def _get_generator_loss(self, real_h, fake_h):
        loss = torch.sum((fake_h - real_h) ** 2, dim=(1, 2))
        
        return loss.mean()

    def _get_discriminator_loss(self, real_logits, fake_logits):
        real_is_real = torch.log(torch.sigmoid(real_logits) + 1e-7)
        fake_is_fake = torch.log(1 - torch.sigmoid(fake_logits) + 1e-7)
        
        return -(real_is_real + fake_is_fake).mean() / 2
    
    def training_step(self, batch, batch_idx):
        target = batch
        self.seq_len = target.shape[1]
        self.batch_size = batch.shape[0]
        generator_optimizer, discriminator_optimizer = self.optimizers()
        noise = torch.randn(self.batch_size, self.seq_len, self.generator.latent_size)
        for _ in range(self.num_disc_steps):
            real_logits, _ = self.discriminator(target)
            with torch.no_grad():
                fake = self.generator(noise)
                
            fake_logits, _ = self.discriminator(fake)
            discriminator_loss = self._get_discriminator_loss(real_logits, fake_logits)
            discriminator_optimizer.zero_grad()
            self.manual_backward(discriminator_loss)
            discriminator_optimizer.step()
            if self.clip_discriminator_weights:
                for dp in self.discriminator.parameters():
                    dp.data.clamp_(-self.clip_discriminator_weights, self.clip_discriminator_weights)

        _, real_h = self.discriminator(target)
        fake = self.generator(noise)
        _, fake_h = self.discriminator(fake)
        generator_loss = self._get_generator_loss(real_h, fake_h)
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        if self.clip_generator_weights:
            for dp in self.generator.parameters():
                dp.data.clamp_(-self.clip_generator_weights, self.clip_generator_weights)

        self.log_dict({
            'train_gen_loss': generator_loss, 
            'train_disc_loss': discriminator_loss
        })

    def configure_optimizers(self):
        generator_optimizer = Adam(self.generator.parameters(), lr=self.generator_lr)
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.discriminator_lr)

        return generator_optimizer, discriminator_optimizer

    def sample(self, seq_len, n_samples):
        z = torch.randn(n_samples, seq_len, self.generator.latent_size)
        with torch.no_grad():
            fake = self.generator(z).cpu()

        return fake
    

class TCNGAN(Generator):
    def __init__(
        self, 
        input_size=3,
        hidden_size=80, 
        latent_size=3,
        num_layers=6, 
        window_size=10, 
        batch_size=16, 
        num_epochs=1,
        verbose=False,
        dropout=0,
        clip_generator_weights=0,
        clip_discriminator_weights=0,
        generator_lr=1e-5,
        discriminator_lr=1e-5,
        num_disc_steps=1,
        weight_norm=False
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.dropout = dropout
        self.clip_generator_weights= clip_generator_weights
        self.clip_discriminator_weights = clip_discriminator_weights
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.num_disc_steps = num_disc_steps
        self.weight_norm = weight_norm

    def fit(self, data):
        super().fit(data)
        self.model = TCNGANModule(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            latent_size=self.latent_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_disc_steps=self.num_disc_steps,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            clip_generator_weights=self.clip_generator_weights,
            clip_discriminator_weights=self.clip_discriminator_weights,
            weight_norm=self.weight_norm
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

