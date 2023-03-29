import torch
from torch import nn

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.l1 = nn.Linear