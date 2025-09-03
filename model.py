import torch
from torch import nn

#The core file of the entire model

class model(nn.Module):
    #temparory in built

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)