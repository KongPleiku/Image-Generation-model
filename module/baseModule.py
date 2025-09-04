import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp



#################################################################
#           Embedding layer for TimeStep and Lebels             #
#################################################################

class TimestepEmbedder(nn.Module):
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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half)/half).to(device=t.device)

        t_float = t
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

class TimestepEmbedder_float16(nn.Module):
    """
    Embeds scalar timesteps into vector representations (this is positional embedding for timesteps t). This use for only float16
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        #setting up the MLP layer to turn the embedding into readable for other module
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=torch.float16),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.float16),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding_float16(t, dim, max_period=10000):
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

        t_float = t.to(dtype=torch.float16)
        t_expanded = torch.unsqueeze(t_float, dim=1)

        freqs_expanded = torch.unsqueeze(freqs, dim=0)
        args = t_expanded * freqs_expanded

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


    def forward(self, t):
        t_freq = self.timestep_embedding_float16(t, self.frequency_embedding_size) #Vectorized timesteps using sinusoidal technique
        t_emb = self.mlp(t_freq) #Pass Embedding to MLP to convert to pass through other modules

        return t_emb

class LabelsEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, hidden_size, num_classes, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0

        if(use_cfg_embedding):
            self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        else:
            self.embedding_table = nn.Embedding(num_classes, hidden_size)

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
    
    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """

        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
