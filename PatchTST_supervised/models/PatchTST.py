__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from layers.enhanced_attention import (
    AutoCorrelationAttention, 
    FourierCrossAttention, 
    ProbAttention,
    ChannelAttention,
    TemporalAttention
)

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # Enhanced attention mechanisms
        self.use_autocorr = getattr(configs, 'use_autocorr', False)
        self.use_fourier = getattr(configs, 'use_fourier', False)
        self.use_channel_attention = getattr(configs, 'use_channel_attention', False)
        self.use_temporal_attention = getattr(configs, 'use_temporal_attention', False)
        self.n_heads = n_heads
        
        # AutoCorrelation for seasonal patterns
        if self.use_autocorr:
            self.autocorr_attention = AutoCorrelationAttention(
                factor=getattr(configs, 'autocorr_factor', 1),
                attention_dropout=attn_dropout,
                output_attention=store_attn
            )
            self.autocorr_proj_in = nn.Linear(c_in, d_model)
            self.autocorr_proj_out = nn.Linear(d_model, c_in)
        
        # Fourier Cross-Attention for frequency patterns
        if self.use_fourier:
            self.fourier_attention = FourierCrossAttention(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=context_window,
                seq_len_kv=context_window,
                modes=getattr(configs, 'fourier_modes', min(32, context_window//2)),
                activation='tanh'
            )
            self.fourier_proj_in = nn.Linear(c_in, d_model)
            self.fourier_proj_out = nn.Linear(d_model, c_in)
        
        # Channel attention mechanism
        if self.use_channel_attention:
            self.channel_attention = ChannelAttention(
                c_in, 
                reduction_ratio=getattr(configs, 'channel_reduction', 16)
            )
        
        # Temporal attention for long-range dependencies
        if self.use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                d_model,
                context_window,
                n_heads,
                dropout
            )
            self.temporal_proj_in = nn.Linear(c_in, d_model)
            self.temporal_proj_out = nn.Linear(d_model, c_in)
        
        # Feature fusion if multiple attentions are used
        num_attention_types = sum([
            self.use_autocorr, self.use_fourier, 
            self.use_channel_attention, self.use_temporal_attention
        ])
        
        if num_attention_types > 1:
            self.feature_fusion = nn.Sequential(
                nn.Linear(c_in * (num_attention_types + 1), c_in * 2),  # +1 for original features
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(c_in * 2, c_in)
            )
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    def apply_enhanced_attention(self, x):
        """Apply AutoCorrelation and/or Fourier attention mechanisms"""
        B, L, C = x.shape
        enhanced_features = []
        original_x = x.clone()
        
        # AutoCorrelation for seasonal patterns
        if self.use_autocorr:
            x_proj = self.autocorr_proj_in(x)
            head_dim = x_proj.size(-1) // self.n_heads
            
            # Reshape for multi-head attention format: [B, L, H, D]
            queries = x_proj.view(B, L, self.n_heads, head_dim)
            keys = x_proj.view(B, L, self.n_heads, head_dim)
            values = x_proj.view(B, L, self.n_heads, head_dim)
            
            autocorr_out, _ = self.autocorr_attention(queries, keys, values, None)
            autocorr_out = autocorr_out.view(B, L, -1)
            seasonal_features = self.autocorr_proj_out(autocorr_out)
            enhanced_features.append(seasonal_features)
        
        # Fourier Cross-Attention for frequency patterns
        if self.use_fourier:
            x_proj = self.fourier_proj_in(x)
            head_dim = x_proj.size(-1) // self.n_heads
            
            # Reshape for multi-head attention format: [B, L, H, D]
            queries = x_proj.view(B, L, self.n_heads, head_dim)
            keys = x_proj.view(B, L, self.n_heads, head_dim)
            values = x_proj.view(B, L, self.n_heads, head_dim)
            
            fourier_out, _ = self.fourier_attention(queries, keys, values, None)
            fourier_out = fourier_out.view(B, L, -1)
            frequency_features = self.fourier_proj_out(fourier_out)
            enhanced_features.append(frequency_features)
        
        # Channel attention
        if self.use_channel_attention:
            channel_enhanced = self.channel_attention(x)
            enhanced_features.append(channel_enhanced)
        
        # Temporal attention  
        if self.use_temporal_attention:
            x_proj = self.temporal_proj_in(x)
            temporal_out, _ = self.temporal_attention(x_proj)
            temporal_features = self.temporal_proj_out(temporal_out)
            enhanced_features.append(temporal_features)
        
        # Combine features
        if len(enhanced_features) == 0:
            return x
        elif len(enhanced_features) == 1:
            return x + 0.1 * enhanced_features[0]  # Small residual weight
        else:
            # Multiple attention mechanisms - use fusion
            all_features = torch.cat([original_x] + enhanced_features, dim=-1)
            fused_features = self.feature_fusion(all_features)
            return fused_features
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        # Apply enhanced attention mechanisms first
        if hasattr(self, 'use_autocorr') and (self.use_autocorr or self.use_fourier or 
                                              self.use_channel_attention or self.use_temporal_attention):
            x = self.apply_enhanced_attention(x)
        
        # Original PatchTST processing
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x
    
    def get_attention_weights(self, x):
        """Extract attention weights for visualization"""
        attention_weights = {}
        
        if self.use_temporal_attention:
            x_proj = self.temporal_proj_in(x)
            _, temporal_weights = self.temporal_attention(x_proj)
            attention_weights['temporal'] = temporal_weights
        
        return attention_weights
    
    def get_model_complexity(self):
        """Calculate model parameters and size"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
        }
    
    @staticmethod
    def validate_config(configs):
        """Validate model configuration"""
        required_attrs = ['enc_in', 'seq_len', 'pred_len', 'e_layers', 'n_heads', 
                         'd_model', 'd_ff', 'dropout', 'patch_len', 'stride']
        
        missing_attrs = [attr for attr in required_attrs if not hasattr(configs, attr)]
        if missing_attrs:
            raise ValueError(f"Missing required config attributes: {missing_attrs}")
        
        # Validation checks
        if configs.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if configs.stride <= 0:
            raise ValueError("stride must be positive")
        if configs.seq_len < configs.patch_len:
            raise ValueError("seq_len must be >= patch_len")
        
        return True