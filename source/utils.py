import argparse
from torch.utils.data import TensorDataset
from source.data import GaussianDataModule
import pickle
import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)  # 1
    np.random.seed(seed)  # 2
    torch.manual_seed(seed)  # 3
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 4


def load_data(args):
    if args["dataset_type"] == "UNIFORM":
        with open("data/uniform.pickle", "rb") as f:
            data = pickle.load(f)
            gaussians = {}

    elif args["dataset_type"] == "NORMAL":
        with open("data/normal.pickle", "rb") as f:
            data = pickle.load(f)
            gaussians = {
                "centroids": [data["mean1"], data["mean2"]],
                "covariances": [data["cov1"], data["cov2"]],
                "weights": [0.5],
            }

    inputs, targets = data["inputs"], data["targets"]

    dataset = TensorDataset(inputs, targets)
    datamodule = GaussianDataModule(dataset, batch_size=args["batch_size"])
    return datamodule, gaussians


def return_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the GaussGan model")

    parser.add_argument(
        "--z_dim",
        type=int,
        default=3,
        help="Dimension of the latent space",
    )
    parser.add_argument(
        "--generator_type",
        type=str,
        default="classical_normal",
        help="Type of generator to use ('classical_uniform', 'classical_normal', 'quantum_samples', 'quantum_shadows')",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs to train",
    )
    parser.add_argument(
        "--grad_penalty",
        type=float,
        default=10,
        help="Gradient penalty regularization factor of Wasserstain GAN",
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="Number of discriminator updates per generator update",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="train",
        help="Stage to run ('train', 'test')",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/",
        help="Path to the checkpoint directory",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="NORMAL",
        help="Distribution of the generator ('NORMAL', 'UNIFORM') ",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="GaussGAN",
        help="Name of the experiment",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",  # This allows multiple values
        default=["IsPositive", "LogLikelihood"],
        help="List of metrics to compute",
    )

    parser.add_argument(
        "--agg_method",
        type=str,
        default="prod",
        help="Aggregation method for the rewards.",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--killer",
        type=bool,
        default=False,
        help="Kill one gaussian",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=100,
        help="Number of validation samples step",
    )
    return parser
