import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=10, activation="relu"):
        super(NeuralNetwork, self).__init__()

        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(),
        }
        act_fn = activations.get(activation.lower(), nn.ReLU())

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn)
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
