import tempfile
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from rdkit import Chem
from torch import nn
import pdb
import warnings


from .metrics import ALL_METRICS
from .nn import SimpleMLP


class GaussGan(LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        predictor,
        optimizer,
        *,
        n_critic=5,
        gradient_penalty=10.0,
        killer=False,
        metrics=[],
        gaussians={},
        validation_samples=1000,
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
        # self.dataset = dataset
        self.generator = generator
        self.discriminator = discriminator
        self.predictor = predictor
        self.optimizer = optimizer
        self.metrics = metrics
        self.killer = killer
        self.validation_samples = validation_samples
        self.n_critic = n_critic
        self.gradient_penalty = gradient_penalty
        self.gaussians = gaussians

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
        # data, labels = batch

        # train discriminator
        d_optim.zero_grad()
        d_loss = self._compute_discriminator_loss(batch)
        self.manual_backward(d_loss)
        d_optim.step()
        self.log(
            "DiscriminatorLoss",
            d_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch[0].size(0),
            sync_dist=True,
        )

        # train predictor
        if self.killer == True:
            p_loss, p_aux = self._compute_predictor_loss(batch)
            self.manual_backward(p_loss)
            p_optim.step()
            p_optim.zero_grad()
            self.log(
                "PredictorLoss",
                p_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch[0].size(0),
                sync_dist=True,

            )
            for key, value in p_aux.items():
                self.log(
                    key,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch[0].size(0),
                    sync_dist=True,
                )

        if (batch_idx % self.n_critic) == 0:
            # train generator
            g_optim.zero_grad()
            g_loss = self._compute_generator_loss(batch)
            self.manual_backward(g_loss)
            g_optim.step()
            self.log(
                "GeneratorLoss",
                g_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch[0].size(0),
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):

        fake_data = self._generate_fake_data(self.validation_samples).detach()

        # Compute and log metrics on generated data
        metrics_fake = self._compute_metrics(fake_data)
        avg_metrics_fake = {
            f"ValidationStep_FakeData_{k}": np.mean(
                [val for val in v if val is not None]
            )
            for k, v in metrics_fake.items()
        }
        self.log_dict(avg_metrics_fake, on_epoch=True, on_step=False, prog_bar=True,  logger=True,                 batch_size=batch[0].size(0),
                sync_dist=True)

        csv_string = "x,y\n" + "\n".join([f"{row[0]},{row[1]}" for row in fake_data])
        try:
            # Attempt to log CSV file as an artifact if logger supports it
            self.logger.experiment.log_text(
                text=csv_string,
                artifact_file=f"gaussian_generated_epoch_{self.current_epoch:04d}.csv",
                run_id=self.logger.run_id,
            )
        except AttributeError:
            print("Could not log the CSV file as an artifact.")

        fig, ax = plt.subplots()
        ax.scatter(fake_data[:, 0].cpu().numpy(), fake_data[:, 1].cpu().numpy())
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title(f"Epoch {self.current_epoch + 1}")

        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile.name)
            plt.close(fig)
            # Log the image file as an artifact if logger supports it
            try:
                img = Image.open(tmpfile.name)
                self.logger.experiment.log_image(
                    image=img,
                    artifact_file=f"scatter_plot_epoch_{self.current_epoch:04d}.png",
                    run_id=self.logger.run_id,
                )
            except AttributeError:
                print("Could not log the image file as an artifact.")

        return fake_data

    def _generate_fake_data(self, batch_size):
        fake_gaussians = self.generator(batch_size)
        fake_gaussians = fake_gaussians.to(self.device)
        return fake_gaussians

    def _apply_discriminator(self, x):
        return self.discriminator(x)

    def _apply_predictor(self, x):
        return torch.sigmoid(self.predictor(x))

    def _compute_discriminator_loss(self, batch):

        fake_gaussians = self._generate_fake_data(batch[0].size(0))
        fake_gaussians = fake_gaussians.detach()
        d_fake = self._apply_discriminator(fake_gaussians)
        d_real = self._apply_discriminator(batch[0])

        # thanks cedric villani
        d_loss = d_fake.mean() - d_real.mean()

        # Compute gradient penalty
        eps = torch.rand(batch[0].size(0), 1).to(self.device)
        fake_inter = (eps * batch[0] + (1.0 - eps) * fake_gaussians).requires_grad_(
            True
        )
        d_inter = self._apply_discriminator(fake_inter)
        gradient_penalty = self._calculate_gradient_penalty(d_inter, fake_inter)
        d_loss += self.gradient_penalty * gradient_penalty
        return d_loss

    def _compute_generator_loss(self, batch):
        x_fake = self._generate_fake_data(batch[0].size(0))
        d_fake = self._apply_discriminator(x_fake)
        gan_loss = -d_fake.mean()

        if self.killer == True:
            rl_loss = -self._apply_predictor(x_fake).mean()
        else:
            rl_loss = 0

        g_loss = gan_loss + rl_loss
        return g_loss

    def _compute_predictor_loss(self, batch):
        x, r = batch
        v = self._apply_predictor(x)
        p_loss = nn.HuberLoss()(v, r)
        return p_loss, {"Predictor_loss": p_loss}

    def _compute_metrics(self, batch):
        metrics = {}
        for metric in self.metrics:
            if metric == "LogLikelihood":
                metrics[metric] = ALL_METRICS[metric](
                    centroids=self.gaussians["centroids"],
                    cov_matrices=self.gaussians["covariances"],
                    weights=self.gaussians["weights"],
                ).compute_score(batch)
            else:
                metrics[metric] = ALL_METRICS[metric]().compute_score(batch)
        return metrics

class LightningSimpleMLP(LightningModule):
    def __init__(self, input_dim, n_layers, hidden_dim, output_dim, dropout, lr):
        super().__init__()
        self.model = SimpleMLP(input_dim, n_layers, hidden_dim, output_dim, dropout)
        self.lr = lr
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters()), []

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)