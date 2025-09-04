import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp



#################################################################
#           Embedding layer for TimeStep and Lebels             #
#################################################################

class TimestepEmbedder(nn.modules):
    """
    Embeds scalar timesteps into vector representations (this is positional embedding for timesteps t).
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        #setting up the MLP layer to turn the embedding into readable for other module
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float16)/half).to(device=t.device)

        t_float = t.to(torch.float16)
        t_expanded = torch.unsqueeze(t_float, dim=1)

        freqs_expanded = torch.unsqueeze(freqs, dim=0)
        args = t_expanded * freqs_expanded

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size) #Vectorized timesteps using sinusoidal technique
        t_emb = self.mlp(t_freq) #Pass Embedding to MLP to convert to pass through other modules

        return t_emb

class LabelsEmbedder(nn.modules):
    def __init__():
        pass

    def forward():
        pass
