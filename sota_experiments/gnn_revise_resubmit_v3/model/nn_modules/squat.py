# modified from https://github.com/rowanz/neural-motifs
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy

def set_diff(a, b):
    combined = torch.cat((a, b))
    uniques, counts = combined.unique(return_counts=True)
    diff = uniques[counts == 1]
    
    return diff

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )
    
    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=0, keepdim=True).expand(z_local.shape[0], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out

    
class P2PDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, unary_output=False):
        super(P2PDecoder, self).__init__() 
        if num_layers == 0:
            self.layers = []
        else:
            self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers 
        self.norm = norm
        self.unary_output = unary_output
        
    def forward(self, tgt, memory, ind, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        pair = tgt 
        if self.num_layers == 0: 
            if self.unary_output:
                return memory[0], tgt 
            else: return tgt
        
        for mod in self.layers:
            unary, pair = mod(pair, memory, ind, tgt_mask=tgt_mask,
                               memory_mask=memory_mask, 
                               tgt_key_padding_mask=tgt_key_padding_mask, 
                               memory_key_padding_mask=memory_key_padding_mask)
            
        if self.norm is not None:
            pair = self.norm(pair)
            unary = self.norm(unary)
            
        if self.unary_output: 
            return unary, pair
        
        return pair
    
    
class P2PDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, norm_first=False):
        super(P2PDecoderLayer, self).__init__() 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_node = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_e2e  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_e2n = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_n2e  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_n2n = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.linear1_unary = nn.Linear(d_model, dim_feedforward)
        self.dropout_unary = nn.Dropout(dropout)
        self.linear2_unary = nn.Linear(dim_feedforward, d_model)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm1_unary = nn.LayerNorm(d_model)
        self.norm2_unary = nn.LayerNorm(d_model)
        self.norm3_unary = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout1_unary = nn.Dropout(dropout)
        self.dropout2_unary = nn.Dropout(dropout)
        self.dropout3_unary = nn.Dropout(dropout)
        
        self.activation = activation
        
    def forward(self, tgt, memory, ind, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        sparsified_pair = tgt
        sparsified_unary, entire_pair = memory 
        ind_pair, ind_e2e, ind_n2e = ind 
        
        sparsified_pair = self.norm1(sparsified_pair + self._sa_block(sparsified_pair, None, None))
        sparsified_unary = self.norm1_unary(sparsified_unary + self._sa_node_block(sparsified_unary, None, None))
        
        pair = torch.zeros_like(entire_pair)
        ind_ = torch.logical_not((torch.arange(pair.size(0), device=entire_pair.device).unsqueeze(1) == ind_pair).any(1))
        
        pair[ind_pair] = sparsified_pair
        pair[ind_] = entire_pair[ind_]
        pair_e2e = pair[ind_e2e]
        pair_n2e = pair[ind_n2e]
        
        updated_pair = self.norm2(sparsified_pair + self._mha_e2e(sparsified_pair, pair_e2e, None, None) \
                                                  + self._mha_e2n(sparsified_pair, sparsified_unary, None, None)) 
        updated_pair = self.norm3(updated_pair + self._ff_block_edge(updated_pair)) 
        
        updated_unary = self.norm2(sparsified_unary + self._mha_n2e(sparsified_unary, pair_n2e, None, None) \
                                                    + self._mha_n2n(sparsified_unary, sparsified_unary, None, None)) 
        updated_unary = self.norm3(updated_unary + self._ff_block_node(updated_unary)) 
        
        return updated_unary, updated_pair

    def _sa_block(self, x, attn_mask, key_padding_mask): 
        x = self.self_attn(x, x, x, attn_mask=attn_mask, 
                           key_padding_mask=key_padding_mask, 
                           need_weights=False)[0]
        
        return self.dropout1(x)

    def _sa_node_block(self, x, attn_mask, key_padding_mask): 
        x = self.self_attn_node(x, x, x, attn_mask=attn_mask, 
                                key_padding_mask=key_padding_mask, 
                                need_weights=False)[0]
        
        return self.dropout1_unary(x)
    
    def _mha_e2n(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn_e2n(x, mem, mem, 
                                      attn_mask=attn_mask, 
                                      key_padding_mask=key_padding_mask, 
                                      need_weights=False)[0]
        
        return self.dropout2(x)
    
    def _mha_e2e(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn_e2e(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=False)[0]
        
        return self.dropout2(x)
    
    def _mha_n2e(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn_n2e(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=False)[0]
        
        return self.dropout2_unary(x)
    
    def _mha_n2n(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn_n2n(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=False)[0]
        
        return self.dropout2_unary(x)

    def _ff_block_node(self, x): 
        x = self.linear2_unary(self.dropout_unary(self.activation(self.linear1_unary(x))))
        return self.dropout3_unary(x)
    
    def _ff_block_edge(self, x): 
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
        
        
class SquatContext(nn.Module):
    def __init__(self, config, hidden_dim=512, num_iter=3):
        super(SquatContext, self).__init__()

        self.cfg = config
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter


        self.pooling_dim = self.cfg['message_size']

        norm_first = True
        
        decoder_layer = P2PDecoderLayer(self.pooling_dim, 8, self.hidden_dim * 2, norm_first=norm_first)
        
        num_layer = 3
        
        self.m2m_decoder = P2PDecoder(decoder_layer, num_layer)
        
        self.mask_predictor = MaskPredictor(self.pooling_dim, self.hidden_dim)
        self.mask_predictor_e2e = MaskPredictor(self.pooling_dim, self.hidden_dim)
        self.mask_predictor_n2e = MaskPredictor(self.pooling_dim, self.hidden_dim) 
        
        self.rho = 0.7
        self.beta = 0.7

    def set_pretrain_pre_clser_mode(self, val=True):
        self.pretrain_pre_clser_mode = val

    def forward(self, obj_feat, feat_pred, num_obj):

        feat_pred_batch_key = []
        
        batch_size = feat_pred.size(0)
        max_num_obj = obj_feat.size(1)

        for k in range(batch_size):
            
            try:
                temp_num_obj = int(num_obj[k])
            except:
                temp_num_obj = int(num_obj[k].item())

            temp = feat_pred[k, :temp_num_obj, :temp_num_obj]
            feat_pred_batch_key.append(temp.flatten(0, 1))


        masks = [self.mask_predictor(k).squeeze(1) for k in feat_pred_batch_key] # num_rel X 1 
        top_inds = [torch.topk(mask, int(mask.size(0) * self.rho))[1] for mask in masks]
        feat_pred_batch_query = [k[top_ind] for k, top_ind in zip(feat_pred_batch_key, top_inds)]
        
        masks_e2e = [self.mask_predictor_e2e(k).squeeze(1) for k in feat_pred_batch_key]
        top_inds_e2e = [torch.topk(mask, int(mask.size(0) * self.beta))[1] for mask in masks_e2e]
        
        masks_n2e = [self.mask_predictor_n2e(k).squeeze(1) for k in feat_pred_batch_key]
        top_inds_n2e = [torch.topk(mask, int(mask.size(0) * self.beta))[1] for mask in masks_n2e]
        

        augment_obj_feat = []

        for k in range(batch_size):
            
            temp_num_obj = int(num_obj[k].item())
            temp = obj_feat[k, :temp_num_obj]
            augment_obj_feat.append(temp)

        # q_ = feat_pred_batch_key[0].unsqueeze(1)
        # u_ = augment_obj_feat[0].unsqueeze(1)
        # v_ = feat_pred_batch_query[0].unsqueeze(1)
        # (ind, ind_e2e, ind_n2e) = (top_inds[0], top_inds_e2e[0], top_inds_n2e[0])
        # r = self.m2m_decoder(v_, (u_,q_), (ind, ind_e2e, ind_n2e))
  
        feat_pred_batch = [self.m2m_decoder(q.unsqueeze(1), (u.unsqueeze(1), p.unsqueeze(1)), (ind, ind_e2e, ind_n2e)).squeeze(1) \
                           for p, u, q, ind, ind_e2e, ind_n2e in \
                           zip(feat_pred_batch_key, augment_obj_feat, feat_pred_batch_query, top_inds, top_inds_e2e, top_inds_n2e)]
        
        entire_sets = [set(range(mask.size(0))) for mask in masks]
            
        feat_pred_batch_ = []
        for idx, (top_ind, k, out) in enumerate(zip(top_inds, feat_pred_batch_key, feat_pred_batch)):
            remaining_ind = set_diff(torch.arange(k.size(0), device=k.device), top_ind)
            feat_pred_ = torch.zeros_like(k)
            feat_pred_[top_ind] = out 
            feat_pred_[remaining_ind] = k[remaining_ind]

            feat_pred_batch_.append(feat_pred_)

        feat_pred_ = torch.cat(feat_pred_batch_, dim=0)
        
        feat_dim = feat_pred_.size(-1)
        return_tensor = torch.zeros((batch_size, max_num_obj, max_num_obj, feat_dim), device=feat_pred_.device)
        current = 0
        
        for b in range(batch_size):
            
            temp_num_obj = int(num_obj[b].item())
            temp_tensor = feat_pred_[current:current + (temp_num_obj**2), :]
            temp_tensor = temp_tensor.view(temp_num_obj, temp_num_obj, feat_dim)
            
            return_tensor[b, :temp_num_obj, :temp_num_obj, :] = temp_tensor
            current += (temp_num_obj**2)
        
        return return_tensor, (masks, masks_e2e, masks_n2e)

    def set_pretrain_pre_clser_mode(self, val=True):
        self.pretrain_pre_clser_mode = val