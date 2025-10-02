"""
Enhanced Attention Mechanisms for PatchTST
Includes AutoCorrelation, Fourier Cross-Attention, and other advanced attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AutoCorrelationAttention(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This mechanism can replace the self-attention family.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelationAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is training mode.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
                    .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)\
                       .repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr)\
                .permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr)\
                .permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class FourierCrossAttention(nn.Module):
    """
    Simplified Fourier Cross Attention mechanism
    """
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, 
                 mode_select_method='random', activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print('fourier enhanced cross attention used!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = min(modes, seq_len_q // 4)  # Conservative mode selection
        self.activation = activation
        
        print(f'Fourier modes: {self.modes}, seq_len: {seq_len_q}')
        
        # Simple learnable frequency weights
        self.freq_weights = nn.Parameter(
            torch.randn(self.modes, dtype=torch.cfloat) * 0.01
        )
        
        # Projection layers
        self.in_projection = nn.Linear(in_channels, out_channels)
        self.out_projection = nn.Linear(out_channels, out_channels)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        
        # Combine heads for processing
        x = q.view(B, L, H * E)  # [B, L, H*E]
        
        # Project input
        x_proj = self.in_projection(x)  # [B, L, out_channels]
        
        # Apply Fourier transform
        x_freq = torch.fft.rfft(x_proj, dim=1)  # [B, L//2+1, out_channels]
        
        # Apply learnable frequency weights to selected modes (avoid in-place operation)
        n_modes = min(self.modes, x_freq.size(1))
        if n_modes > 0:
            # Create a copy to avoid in-place modification
            x_freq_weighted = x_freq.clone()
            freq_weights_expanded = self.freq_weights[:n_modes].unsqueeze(0).unsqueeze(-1)
            x_freq_weighted[:, :n_modes, :] = x_freq[:, :n_modes, :] * freq_weights_expanded
            x_freq = x_freq_weighted
        
        # Inverse Fourier transform
        x_time = torch.fft.irfft(x_freq, n=L, dim=1)  # [B, L, out_channels]
        
        # Final projection
        output = self.out_projection(x_time)  # [B, L, out_channels]
        
        # Reshape back to multi-head format
        output = output.view(B, L, H, self.out_channels // H)
        
        return (output, None)


class ProbAttention(nn.Module):
    """
    ProbSparse Attention mechanism from Informer
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


class ChannelAttention(nn.Module):
    """Channel attention mechanism to learn inter-channel dependencies"""
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        
        # Global average pooling and max pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Shared MLP for both pooling operations
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [Batch, Length, Channel]
        batch_size, length, channels = x.shape
        
        # Transpose for pooling operations: [Batch, Channel, Length]
        x_permuted = x.transpose(1, 2)
        
        # Global average pooling and max pooling
        avg_pool = self.avg_pool(x_permuted).squeeze(-1)  # [Batch, Channel]
        max_pool = self.max_pool(x_permuted).squeeze(-1)  # [Batch, Channel]
        
        # Apply shared MLP
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        
        # Channel attention weights
        channel_attention = self.sigmoid(avg_out + max_out)  # [Batch, Channel]
        
        # Apply attention weights
        channel_attention = channel_attention.unsqueeze(-1)  # [Batch, Channel, 1]
        attended = x_permuted * channel_attention
        
        return attended.transpose(1, 2)  # Back to [Batch, Length, Channel]


class TemporalAttention(nn.Module):
    """Temporal attention for capturing long-range dependencies"""
    def __init__(self, d_model, seq_len, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        
        # Multi-head self-attention for temporal dependencies
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal convolution for local patterns
        self.temporal_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model  # Depthwise convolution
        )
        
        # Position-wise feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [Batch, Length, d_model]
        
        # Apply temporal convolution for local patterns
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Self-attention for global temporal dependencies
        residual = x
        x = self.layer_norm1(x + x_conv)
        
        attended, attention_weights = self.temporal_attention(x, x, x)
        x = residual + self.dropout(attended)
        
        # Feedforward
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.dropout(self.feedforward(x))
        
        return x, attention_weights