import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from rdkit import Chem
from torch import nn
import pdb
import warnings

from .metrics import ALL_METRICS

class GaussGan(LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        predictor,
        optimizer,
        *,
        n_critic=5,
        grad_penalty=10.0,
        process_method="soft_gumbel",
        agg_method="prod",
        metrics=["being_right"],
    ):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic optimization
        self.save_hyperparameters(
            ignore=[
                "generator",
                "discriminator",
                "predictor",
                "optimizer",
            ],
        )
        #self.dataset = dataset
        self.generator = generator
        self.discriminator = discriminator
        self.predictor = predictor
        self.optimizer = optimizer
        self.metrics = metrics

    def configure_optimizers(self):
        g_optim = self.optimizer(self.generator.parameters())
        d_optim = self.optimizer(self.discriminator.parameters())
        p_optim = self.optimizer(self.predictor.parameters())
        return [g_optim, d_optim, p_optim], []


    def _calculate_gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def training_step(self, batch, batch_idx):
        # Access the optimizers
        g_optim, d_optim, p_optim = self.optimizers()

        # train discriminator
        d_loss = self._compute_discriminator_loss(batch)
        self.manual_backward(d_loss)
        d_optim.step()
        d_optim.zero_grad()
        self.log(
            "Discriminator Loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.features["X"].size(0),
        )

        # train predictor
        p_loss, p_aux = self._compute_predictor_loss(batch)
        self.manual_backward(p_loss)
        p_optim.step()
        p_optim.zero_grad()
        self.log(
            "Predictor_loss",
            p_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.size(0),
        )
        for key, value in p_aux.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.size(0),
            )

        if (batch_idx % self.hparams.n_critic) == 0:
            # train generator
            g_loss = self._compute_generator_loss(batch)
            self.manual_backward(g_loss)
            g_optim.step()
            g_optim.zero_grad()
            self.log(
                "Generator_Loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.size(0),
            )

    def validation_step(self, batch, batch_idx):
        # Similar to test_step but for validation data
        # Process the real data
        #pdb.set_trace()
        # Generate fake data
        fake_data = self._generate_fake_data(batch[0].size(0))
        # Compute metrics on generated data
        metrics_fake = self._compute_metrics(fake_data)
        # avg_metrics_fake = {
        #     f"Validation_step_fake_data_{k}": np.mean(v) for k, v in metrics_fake.items()
        # }


        avg_metrics_fake = {
            f"Validation_step_fake_data_{k}": np.mean(
                [val for val in v if val is not None]
            )
            for k, v in metrics_fake.items()
        }

        # Log the metrics
        self.log_dict(avg_metrics_fake, on_epoch=True, prog_bar=True, logger=True)

        # Optionally, compute an aggregated validation metric
        # val_metric = np.mean(list(avg_metrics_fake.values()))
        # self.log(
        #     "Aggregated_metric_during_validation",
        #     val_metric,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        # Log the points as csv files


        csv_string = "x,y\n" + "\n".join([f"{row[0]},{row[1]}" for row in fake_data])
        try:
            # Attempt to log CSV file as an artifact if logger supports it
            self.logger.experiment.log_text(
                text=csv_string,
                artifact_file=f"gaussian_generated_epoch_{self.current_epoch:04d}.csv",
                run_id=self.logger.run_id,
            )
        except AttributeError:
            warnings.warn("Could not log the CSV file as an artifact.")

        return fake_data

    def _generate_fake_data(self, batch_size):
        fake_gaussians = self.generator(batch_size)
        fake_gaussians = fake_gaussians.to(self.device)
        return fake_gaussians

    def _apply_discriminator(self, x):
        return self.discriminator(x)

    def _apply_predictor(self, a, x):
        return torch.sigmoid(self.predictor(x))

    def _compute_discriminator_loss(self, batch):
        
        fake_gaussians = self._generate_fake_data(batch[0].size(0))
        fake_gaussians = fake_gaussians.detach()
        d_fake = self._apply_discriminator(fake_gaussians)
        d_real = self._apply_discriminator(batch[0])
        d_loss = d_fake.mean() - d_real.mean()

        # Compute gradient penalty
        eps = torch.rand(batch.size(0),1).to(self.device)
        fake_inter = (eps * fake_gaussians + (1.0 - eps) * fake_gaussians).requires_grad_(True)
        d_inter = self._apply_discriminator(fake_inter)
        grad_penalty = self._calculate_gradient_penalty(d_inter, fake_inter)
        d_loss += self.hparams.grad_penalty * grad_penalty
        return d_loss

    def _compute_generator_loss(self, batch):
        a_fake, x_fake = self._generate_fake_data(batch[0].size(0))
        a_fake, x_fake = self._process_fake_data(a_fake, x_fake)
        d_fake = self._apply_discriminator(a_fake, x_fake)
        gan_loss = -d_fake.mean()
        rl_loss = -self._apply_predictor(a_fake, x_fake).mean()
        g_loss = gan_loss + rl_loss
        return g_loss

    def _compute_predictor_loss(self, batch):
        x, r = batch
        v = self._apply_predictor(x)
        p_loss = nn.HuberLoss()(v, r)
        return p_loss


    def _compute_metrics(self, batch):
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = ALL_METRICS[metric]().compute_score(batch)
        return metrics