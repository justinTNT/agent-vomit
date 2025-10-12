import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    def __init__(self, d_model_1, d_model_2, d_hidden, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_hidden % n_heads == 0
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_k = d_hidden // n_heads
        
        # Projections from each modality to common space
        self.q_proj_1 = nn.Linear(d_model_1, d_hidden)
        self.k_proj_2 = nn.Linear(d_model_2, d_hidden)
        self.v_proj_2 = nn.Linear(d_model_2, d_hidden)
        
        # Reverse direction
        self.q_proj_2 = nn.Linear(d_model_2, d_hidden)
        self.k_proj_1 = nn.Linear(d_model_1, d_hidden)
        self.v_proj_1 = nn.Linear(d_model_1, d_hidden)
        
        # Output projections back to original dimensions
        self.out_proj_1 = nn.Linear(d_hidden, d_model_1)
        self.out_proj_2 = nn.Linear(d_hidden, d_model_2)
        
        self.dropout = nn.Dropout(dropout)
        
    def attend(self, query, key, value, proj_q, proj_k, proj_v, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)
        
        # Project and reshape
        Q = proj_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = proj_k(key).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = proj_v(value).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask == 0, -1e9)
            else:
                scores = scores + mask
                
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_hidden
        )
        
        return attn_output, attn_weights
    
    def forward(self, modal_1, modal_2, mask_1=None, mask_2=None):
        # Modal 1 attending to Modal 2
        attn_1_to_2, weights_1_to_2 = self.attend(
            modal_1, modal_2, modal_2,
            self.q_proj_1, self.k_proj_2, self.v_proj_2,
            mask=mask_2
        )
        out_1 = self.out_proj_1(attn_1_to_2)
        
        # Modal 2 attending to Modal 1
        attn_2_to_1, weights_2_to_1 = self.attend(
            modal_2, modal_1, modal_1,
            self.q_proj_2, self.k_proj_1, self.v_proj_1,
            mask=mask_1
        )
        out_2 = self.out_proj_2(attn_2_to_1)
        
        return {
            'modal_1_output': out_1,
            'modal_2_output': out_2,
            'attention_1_to_2': weights_1_to_2,
            'attention_2_to_1': weights_2_to_1
        }


class CrossModalFusion(nn.Module):
    def __init__(self, d_model_1, d_model_2, d_hidden=512, n_heads=8, 
                 n_layers=4, dropout=0.1, fusion_type='cross_attention',
                 output_dim=None):
        super().__init__()
        self.d_model_1 = d_model_1
        self.d_model_2 = d_model_2
        self.d_hidden = d_hidden
        self.fusion_type = fusion_type
        self.output_dim = output_dim or d_hidden
        
        # Project to common dimension if needed
        self.proj_1 = nn.Linear(d_model_1, d_hidden) if d_model_1 != d_hidden else nn.Identity()
        self.proj_2 = nn.Linear(d_model_2, d_hidden) if d_model_2 != d_hidden else nn.Identity()
        
        if fusion_type == 'cross_attention':
            # Stack of cross-modal attention layers
            self.fusion_layers = nn.ModuleList([
                CrossModalAttention(d_hidden, d_hidden, d_hidden, n_heads, dropout)
                for _ in range(n_layers)
            ])
            
            # Layer norms for each modality
            self.norm_1 = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
            self.norm_2 = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
            
        elif fusion_type == 'concatenate':
            # Simple concatenation with MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_hidden * 2, d_hidden * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden * 2, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
        elif fusion_type == 'multiplicative':
            # Multiplicative fusion (like in VQA models)
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_hidden * 2, d_hidden),
                nn.Sigmoid()
            )
            self.fusion_transform = nn.Linear(d_hidden, d_hidden)
            
        elif fusion_type == 'bottleneck':
            # Bottleneck fusion through lower dimension
            bottleneck_dim = d_hidden // 4
            self.down_proj_1 = nn.Linear(d_hidden, bottleneck_dim)
            self.down_proj_2 = nn.Linear(d_hidden, bottleneck_dim)
            self.fusion_bottleneck = nn.Sequential(
                nn.Linear(bottleneck_dim * 2, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, d_hidden)
            )
        
        # Final projection
        self.output_projection = nn.Linear(d_hidden, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, modal_1, modal_2, mask_1=None, mask_2=None, return_all=False):
        # Project to common dimension
        feat_1 = self.proj_1(modal_1)
        feat_2 = self.proj_2(modal_2)
        
        if self.fusion_type == 'cross_attention':
            # Apply cross-attention layers with residual connections
            for i, (fusion_layer, norm1, norm2) in enumerate(zip(self.fusion_layers, self.norm_1, self.norm_2)):
                fusion_out = fusion_layer(feat_1, feat_2, mask_1, mask_2)
                
                # Residual connections and layer norm
                feat_1 = norm1(feat_1 + self.dropout(fusion_out['modal_1_output']))
                feat_2 = norm2(feat_2 + self.dropout(fusion_out['modal_2_output']))
            
            # Combine features (can be task-specific)
            if hasattr(self, 'pool_strategy'):
                fused = self.pool_features(feat_1, feat_2, mask_1, mask_2)
            else:
                # Default: concatenate mean pooled features
                pooled_1 = feat_1.mean(dim=1) if len(feat_1.shape) > 2 else feat_1
                pooled_2 = feat_2.mean(dim=1) if len(feat_2.shape) > 2 else feat_2
                fused = (pooled_1 + pooled_2) / 2
                
        elif self.fusion_type == 'concatenate':
            # Pool if needed
            if len(feat_1.shape) > 2:
                feat_1 = feat_1.mean(dim=1)
            if len(feat_2.shape) > 2:
                feat_2 = feat_2.mean(dim=1)
            
            concat_feats = torch.cat([feat_1, feat_2], dim=-1)
            fused = self.fusion_mlp(concat_feats)
            
        elif self.fusion_type == 'multiplicative':
            # Pool if needed
            if len(feat_1.shape) > 2:
                feat_1 = feat_1.mean(dim=1)
            if len(feat_2.shape) > 2:
                feat_2 = feat_2.mean(dim=1)
                
            concat_feats = torch.cat([feat_1, feat_2], dim=-1)
            gate = self.fusion_gate(concat_feats)
            fused = self.fusion_transform(feat_1 * feat_2) * gate
            
        elif self.fusion_type == 'bottleneck':
            # Pool if needed
            if len(feat_1.shape) > 2:
                feat_1 = feat_1.mean(dim=1)
            if len(feat_2.shape) > 2:
                feat_2 = feat_2.mean(dim=1)
                
            bottle_1 = self.down_proj_1(feat_1)
            bottle_2 = self.down_proj_2(feat_2)
            concat_bottle = torch.cat([bottle_1, bottle_2], dim=-1)
            fused = self.fusion_bottleneck(concat_bottle)
        
        # Final projection
        output = self.output_projection(fused)
        
        if return_all:
            return {
                'fused': output,
                'modal_1_features': feat_1,
                'modal_2_features': feat_2
            }
        else:
            return output