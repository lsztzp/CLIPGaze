import copy
from typing import Optional
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=512, decoder_dropout=0.2,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, device="cuda:0", depth=3, extra_blocks=0, rev_activations=False,
                 cond_layer=(0, 1, 2), reduce_cond=None, im_h=20, im_w=32):
        super().__init__()
        self.device = device

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, decoder_dropout, activation,
                                                normalize_before).to(device)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                     return_intermediate=return_intermediate_dec).to(device)
        dropout = nn.Dropout(decoder_dropout)
        self.decoder = TransformerDecoderWrapper(d_model, activation, decoder, dropout, device)

        self.d_model = d_model
        self.nhead = nhead

        self.rev_activations = rev_activations
        self.cond_layer = cond_layer
        self.reduces = nn.ModuleList([nn.Linear(1024, d_model) for _ in range(depth)])
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead) for _ in range(depth)])
        self.extra_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead) for _ in range(extra_blocks)])
        # conditional
        if reduce_cond is not None:
            self.reduce_cond = nn.Linear(1024, reduce_cond)
            for p in self.reduce_cond.parameters():
                p.requires_grad_(False)
        else:
            self.reduce_cond = None

        self.film_mul = nn.ModuleList(
            [nn.Linear(768 if reduce_cond is None else reduce_cond, d_model) for _ in range(depth)])
        self.film_add = nn.ModuleList(
            [nn.Linear(768 if reduce_cond is None else reduce_cond, d_model) for _ in range(depth)])

        self.patchpos_embed = PositionEmbeddingSine2d((im_h, im_w), hidden_dim=self.d_model, normalize=True,
                                                      device=device)

    def forward(self, src: Tensor, tgt: Tensor, task: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        memory_task = self.feature_merge(src, task)

        output = self.decoder(tgt, memory_task=memory_task, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              querypos_embed=querypos_embed, patchpos_embed=patchpos_embed)
        return output

    def feature_merge(self, activations, cond):
        _activations = activations[::-1] if not self.rev_activations else activations
        assert len(_activations) == len(self.reduces)

        a = None
        for i, (activation, block, reduce) in enumerate(zip(_activations, self.blocks, self.reduces)):
            activation = activation.to(self.device)
            if a is not None:
                a = reduce(activation) + a
            else:
                a = reduce(activation)

            if i in self.cond_layer:
                if self.reduce_cond is not None:
                    cond = self.reduce_cond(cond)

                a = self.film_mul[i](cond) * a + self.film_add[i](cond)

            a[1:] = self.patchpos_embed(a[1:])

            a = block(a)

        for block in self.extra_blocks:
            a = a + block(a)

        a = a[1:]
        return a


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, d_model, activation, decoder, dropout, device):
        super().__init__()
        self.device = device
        self.activation = _get_activation_fn(activation)
        self.decoder = decoder.to(device)
        self.dropout = dropout

        self.linear = nn.Linear(d_model * 2, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos(tensor)

    def forward(self, tgt, memory_task,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):

        memory_task = self.dropout(self.activation(memory_task))

        # decoder
        output = self.decoder(tgt, memory_task, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              querypos_embed=querypos_embed, patchpos_embed=patchpos_embed)
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           querypos_embed=querypos_embed,
                           patchpos_embed=patchpos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     querypos_embed: Optional[Tensor] = None,
                     patchpos_embed: Optional[Tensor] = None):
        q = k = v = self.with_pos_embed(tgt, querypos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # key_tmp = torch.cat((memory[0].unsqueeze(0),patchpos_embed(memory[1:])),0)    #1

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, querypos_embed),
                                   key=patchpos_embed(memory),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    querypos_embed: Optional[Tensor] = None,
                    patchpos_embed: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = v = self.with_pos_embed(tgt2, querypos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, querypos_embed),
                                   key=patchpos_embed(memory),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    querypos_embed=querypos_embed,
                                    patchpos_embed=patchpos_embed)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 querypos_embed=querypos_embed,
                                 patchpos_embed=patchpos_embed)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine2d(nn.Module):
    def __init__(self, spatial_dim, hidden_dim=512, temperature=10000, normalize=False, scale=None, flatten=True,
                 device="cuda:0"):
        super(PositionEmbeddingSine2d, self).__init__()
        self.num_pos_feats = hidden_dim // 2
        normalize = normalize
        self.h, self.w = spatial_dim
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.device = device
        position_y = torch.arange(self.h).unsqueeze(1)
        position_x = torch.arange(self.w).unsqueeze(1)
        if normalize:
            eps = 1e-6
            position_y = position_y / (self.h - 1 + eps) * scale
            position_x = position_x / (self.w - 1 + eps) * scale
        div_term = torch.exp(torch.arange(0, self.num_pos_feats, 2) * (-math.log(temperature) / self.num_pos_feats))
        pe_y = torch.zeros(self.h, 1, self.num_pos_feats)
        pe_x = torch.zeros(1, self.w, self.num_pos_feats)
        pe_y[:, 0, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 0, 1::2] = torch.cos(position_y * div_term)
        pe_x[0, :, 0::2] = torch.sin(position_x * div_term)
        pe_x[0, :, 1::2] = torch.cos(position_x * div_term)
        pe_y = pe_y.repeat(1, self.w, 1)
        pe_x = pe_x.repeat(self.h, 1, 1)
        self.pos = torch.cat((pe_y, pe_x), dim=-1).permute(2, 0, 1)
        if flatten:
            self.pos = self.pos.view(hidden_dim, -1).permute(1, 0).unsqueeze(1)
        else:
            self.pos = self.pos.permute(1, 2, 0)
        del pe_y, pe_x, position_y, position_x

    def forward(self, x):
        return x.to(self.device) + self.pos.to(self.device)
