import torch
from torch import nn
from torch.nn import functional as F

class single_head_attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class multiple_head_attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class linear_MHSA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class TSMA(nn.Module):
    """
    This module is the implementation of the Time-dependent multi-head self attention from DiffiT.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class linear_TSMA(nn.Module):
    """
    This module implement both linear attention from Sana and the TSMA from DiffiT.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)