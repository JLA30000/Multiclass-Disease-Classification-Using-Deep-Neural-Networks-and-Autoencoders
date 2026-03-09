import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int=64, hidden_dims=(256, 128), dropout: float=0.0):
        super().__init__()
        h1, h2 = hidden_dims
        enc_layers = [nn.Linear(input_dim, h1), nn.ReLU()]
        if dropout > 0:
            enc_layers.append(nn.Dropout(dropout))
        enc_layers += [nn.Linear(h1, h2), nn.ReLU()]
        if dropout > 0:
            enc_layers.append(nn.Dropout(dropout))
        enc_layers += [nn.Linear(h2, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)
        dec_layers = [nn.Linear(latent_dim, h2), nn.ReLU()]
        if dropout > 0:
            dec_layers.append(nn.Dropout(dropout))
        dec_layers += [nn.Linear(h2, h1), nn.ReLU()]
        if dropout > 0:
            dec_layers.append(nn.Dropout(dropout))
        dec_layers += [nn.Linear(h1, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
