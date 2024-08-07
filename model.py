import torch
from torch import nn
import math

class SinActivation(nn.Module):
    def __init__(self, omega):
        super(SinActivation, self).__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)

class SIREN(nn.Module):
    def __init__(
        self, 
        n_layers,
        in_ft = 2,
        hidden_ft = 256,
        out_ft = 3,
        omega = 30
    ):
        super(SIREN, self).__init__()
        self.omega = omega
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_ft, hidden_ft))
                continue
            self.layers.append(SinActivation(self.omega))
            out_size = out_ft if i == n_layers - 1 else hidden_ft
            self.layers.append(nn.Linear(hidden_ft, out_size))

        self.init_weights()

    def init_weights(self):
        for i, (n, p) in enumerate(self.layers.named_parameters()):
            if "weight" not in n:
                continue

            in_ft = p.shape[1]
            if i == 0:
                p.data.uniform_(-1 / in_ft, 1 / in_ft)
            else:
                p.data.uniform_(
                    - math.sqrt(6 / in_ft) / self.omega,
                    math.sqrt(6 / in_ft) / self.omega
                )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x