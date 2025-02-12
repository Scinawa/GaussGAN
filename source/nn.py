import random
from abc import ABC, abstractmethod

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch import nn
from torch.nn.functional import one_hot


class QuantumNoise(nn.Module):
    def __init__(self, num_qubits: int = 8, num_layers: int = 3):
        super(QuantumNoise, self).__init__()

        # Register the parameters with the module
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Initialize weights with PyTorch (between -pi and pi) and register as a learnable parameter
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )

        # Initialize the device
        dev = qml.device("default.qubit", wires=num_qubits)

        # Define the quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def gen_circuit(w):
            # random noise as generator input
            z1 = random.uniform(-1, 1)
            z2 = random.uniform(-1, 1)
            # construct generator circuit for both atom vector and node matrix
            for i in range(num_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[l][i + num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        # Define the quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def gen_circuit_with_input(z1, z2):
            w = self.weights
            # construct generator circuit for both atom vector and node matrix
            for i in range(num_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[l][i + num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        self.gen_circuit = gen_circuit
        self.gen_circuit_with_input = gen_circuit_with_input

    def forward(self, batch_size: int):
        sample_list = [
            torch.concat(
                [tensor.unsqueeze(0) for tensor in self.gen_circuit(self.weights)]
            )
            for _ in range(batch_size)
        ]
        noise = torch.stack(tuple(sample_list)).float()
        return noise

class QuantumShadowNoise(nn.Module):
    @staticmethod
    def build_qnode(num_qubits, num_layers, num_basis):

        paulis = [qml.PauliZ, qml.PauliX, qml.PauliY, qml.Identity]
        basis = [
            qml.operation.Tensor(*[random.choice(paulis)(i) for i in range(num_qubits)])
            for _ in range(num_basis)
        ]

        dev = qml.device("default.qubit", wires=num_qubits, shots=300)

        @qml.qnode(dev, interface="torch", diff_method="best")
        def gen_circuit(w):
            z1 = random.uniform(-1, 1)
            z2 = random.uniform(-1, 1)
            # construct generator circuit for both atom vector and node matrix
            for i in range(num_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[l][i + num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            return qml.shadow_expval(basis)

        return basis, gen_circuit

    def __init__(
        self,
        z_dim: int,
        *,
        num_qubits: int = 8,
        num_layers: int = 3,
        num_basis: int = 3,
    ):
        super(QuantumShadowNoise, self).__init__()

        # Register the parameters with the module
        self.z_dim = z_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_basis = num_basis

        self.basis, self.gen_circuit = self.build_qnode(
            num_qubits, num_layers, num_basis
        )

        # Initialize weights with PyTorch (between -pi and pi) and register as a learnable parameter
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )
        self.coeffs = nn.Parameter(torch.rand(num_basis, self.z_dim))

    def forward(self, batch_size: int):
        sample_list = [
            torch.cat(
                [tensor.unsqueeze(0) for tensor in self.gen_circuit(self.weights)]
            )
            for _ in range(batch_size)
        ]
        noise = torch.stack(tuple(sample_list)).float()
        noise = torch.matmul(noise, self.coeffs)
        return noise

class ClassicalNoise(nn.Module):
    def __init__(self, z_dim: int, generator_type: str):
        super(ClassicalNoise, self).__init__()
        self.z_dim = z_dim
        self.generator_type = generator_type


    def forward(self, batch_size: int):

        if self.generator_type == "classical_uniform":
            return torch.rand(batch_size, self.z_dim)
        elif self.generator_type == "classical_normal":
            return torch.randn(batch_size, self.z_dim)

class MLPGenerator(nn.Module):
    def __init__(self, hidden_dims, z_dim=8, output_dim=2 ):
        super(MLPGenerator, self).__init__()
        layers = []
        current_dim = z_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.ReLU())
            current_dim = hdim
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)


    def forward(self, z):
        out = self.model(z)
        return out


# class MLPDiscriminator(nn.Module):
#     def __init__(self, hidden_dims, input_dim=2, output_dim=1):
#         super(MLPDiscriminator, self).__init__()
#         layers = []
#         current_dim = input_dim
#         for hdim in hidden_dims:
#             layers.append(nn.Linear(current_dim, hdim))
#             layers.append(nn.Tanh())
#             current_dim = hdim
#         layers.append(nn.Linear(current_dim, output_dim))
#         self.model = nn.Sequential(*layers)


#     def forward(self, z):
#         return self.model(z)

class MLPDiscriminator(nn.Module):
    def __init__(self, hidden_dims, input_dim=2, output_dim=1):
        super(MLPDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),          # Input: 2D vector -> Hidden layer
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),         # Hidden layer
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim),          # Output: Probability
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim, output_dim, dropout=0):
        super(SimpleMLP, self).__init__()
        if n_layers < 1:
            raise ValueError("Number of layers must be at least 1.")
        elif n_layers == 1:
            self.model = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(n_layers-1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)