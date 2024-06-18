#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""

import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message (str): Message for error catch
        actual_size (int): the short size that cannot pass the subsampling
        limit (int): the limit size for subsampling

    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv1dSubsampling1) and size < 5:
        return True, 5
    if isinstance(ins, Conv1dSubsampling2) and size < 5:
        return True, 5
    if isinstance(ins, Conv1dSubsampling3) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling1) and size < 5:
        return True, 5
    if isinstance(ins, Conv2dSubsampling2) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1


class Conv1dSubsampling1(torch.nn.Module):
    """Convolutional 1D subsampling.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling1 object."""
        super(Conv1dSubsampling1, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:1][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv1dSubsampling2(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling2 object."""
        super(Conv1dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:1][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv1dSubsampling3(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/3 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling3 object."""
        super(Conv1dSubsampling3, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:1][:, :, :-4:3]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsamplingFUSED_14(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, odim, 3, 2),
                nn.ReLU(),
                nn.Conv2d(odim, odim, 3, 2),
                nn.ReLU(),
            ) for _ in range(4)
        ])
        
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )
        self.fusion_weights = nn.Parameter(torch.ones(4))

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x_mask = x_mask #All the inputs have same dimensions
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        x = torch.stack(x, dim=0)  # (4, b, c, t, f)
        #print(x.shape)
        x = torch.sum(self.fusion_weights.view(-1, 1, 1, 1, 1) * x, dim=0)  # (b, c, t, f)
        #print(x.shape)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


#Second approach
class Conv2dSubsamplingFUSED_15(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, odim, 3, 2),
                nn.ReLU(),
                nn.Conv2d(odim, odim, 3, 2),
                nn.ReLU(),
            ) for _ in range(4)
        ])
        
        self.out = nn.Sequential(
            nn.Linear( 4*(odim * (((idim - 1) // 2 - 1) // 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )
        #self.fusion_weights = nn.Parameter(torch.ones(4))

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x_mask = x_mask #All the inputs have same dimensions
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        x = torch.cat(x, dim=-1)  # (4, b, c, t, f)
        #print(x.shape)
        #x = torch.sum(self.fusion_weights.view(-1, 1, 1, 1, 1) * x, dim=0)  # (b, c, t, f)
        #print(x.shape)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]



class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim,activation='relu'):
        super(Expert, self).__init__()
       
        if activation=='relu':            
            self.activation = nn.ReLU()
        if activation=='gelu':
            self.activation = nn.GELU()
        if activation=='glu':
            self.activation=GEGLU()
        
        if activation=='glu':
            self.fc2 = nn.Linear(hidden_dim//2, input_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        #return self.fc2(self.activation(self.fc1(x)))
        return self.fc2(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return self.fc(x)

#My implementation of Multi Headed Mixture of experts layer

class SMoE_MHA_enc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts,top_k, num_heads,activation):
        super(SMoE_MHA_enc, self).__init__()
        # Input dim is here actually the embedding dimension.
        self.h = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(input_dim // self.h, hidden_dim,activation) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim // self.h, self.num_experts)
        self.pre_split_projection = nn.Linear(input_dim, input_dim)
        self.align = nn.Linear(input_dim, 256)
        self.embedding_dim = input_dim
    
    def forward(self, x):
        # Hardcode num_heads

        # x in shape (Batch, Time, Emb)
        batch_size, seq_length, embedding_dim = x.size()
        assert embedding_dim == self.embedding_dim, "Embedding dimension mismatch"

        # Calculate new embedding dimension per head
        head_dim = embedding_dim // self.h
        assert head_dim * self.h == embedding_dim, "Embedding dimension must be divisible by number of heads"

        # Step 0: Project
        x = self.pre_split_projection(x)
        
        # Step 1: Reshape to [batch_size, num_heads, seq_length, head_dim]
        x = x.view(batch_size, seq_length, self.h, head_dim).permute(0, 2, 1, 3)
        
        # Step 2: Flatten the last two dimensions for gating network input
        x_flat = x.contiguous().view(batch_size * self.h, seq_length, head_dim)
        
        # Step 3: Gating network
        gate_scores = self.gating_network(x_flat)

        topk_scores, topk_indices = torch.topk(softmax_scores, self.top_k, dim=-1)

        # Step 4: Compute outputs for all experts
        all_expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=2)  # (batch_size*num_heads, seq_length, num_experts, head_dim)
        
        # Step 5: Select top-k expert outputs
        selected_expert_outputs = all_expert_outputs.gather(2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))  # (batch_size*num_heads, seq_length, 2, head_dim)
        
        # Step 6: Calculate weights and apply them
        weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # (batch_size*num_heads, seq_length, 2, 1)
        expert_outputs = (selected_expert_outputs * weights).sum(dim=2)  # (batch_size*num_heads, seq_length, head_dim)
        
        # Step 7: Apply residual connection
        x_flat = x_flat + expert_outputs

        # Step 8: Reshape back to original dimensions
        x = x_flat.view(batch_size, self.h, seq_length, head_dim).permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, seq_length, embedding_dim)
        
        # Step 9: Apply FFN to align mis-paired values
        x = self.align(x)
        
        return x







#third approach
class Conv2dSubsamplingFUSED(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, odim, 3, 2),
                nn.ReLU(),
                nn.Conv2d(odim, odim, 3, 2),
                nn.ReLU(),
            ) for _ in range(4)
        ])
        
        self.out = pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        
        
        self.moe_network = SMoE_MHA_enc(1024*76, 1024*76, 8,8, 4,'relu') #num_experts 8, top_k=8 ...I can sacrifice run speed as long as I am in encoder.
    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x_mask = x_mask #All the inputs have same dimensions
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        x = torch.cat(x, dim=-1)  # ( b, c, t, 4*f)
        

        b, c, t, f = x.size()
        x = self.moe_network(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x = self.out(x)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]





class Conv2dSubsampling1(torch.nn.Module):
    """Similar to Conv2dSubsampling module, but without any subsampling performed.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling1 object."""
        super(Conv2dSubsampling1, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (idim - 4), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Pass x through 2 Conv2d layers without subsampling.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim).
                where time' = time - 4.
            torch.Tensor: Subsampled mask (#batch, 1, time').
                where time' = time - 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-4]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
