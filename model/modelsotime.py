import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import numpy as np
import logging
import math
EPS = 1e-15
objective='soft-boundary'
from torch import Tensor



class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3, bias=True)
        self.act = torch.nn.LeakyReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return h

class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)
        # self.act = torch.nn.Tanh

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)


        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.SELU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):

        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2)
        hn = seq_x.mean(dim=1)
        output = self.merger(hn, src_x)
        return output, None


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)#reshape
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class MapBasedMultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        q = torch.unsqueeze(q, dim=2)
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        k = torch.unsqueeze(k, dim=1)
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3])

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        mask = mask.repeat(n_head, 1, 1)

        # Map based Attention
        q_k = torch.cat([q, k], dim=3)
        attn = self.weight_map(q_k).squeeze(dim=3)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())


    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, edge_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.logger = logging.getLogger(__name__)

        self.edge_in_dim = feat_dim
        self.model_dim = self.edge_in_dim
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim)

        self.line = torch.nn.Linear(self.model_dim, feat_dim,bias = False)
        assert (self.model_dim % n_head == 0)
        self.attn_mode = attn_mode
        self.act = torch.nn.LeakyReLU()

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                                                d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head,
                                                                dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t,  mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1)  # src
        q = src_ext
        k = seq

        mask = torch.unsqueeze(mask, dim=2)
        mask = mask.permute([0, 2, 1])

        # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        attn = attn.squeeze()
        
        output = self.merger(output, q.squeeze())
        return output, attn

class TGAN(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat, num_layers=3, use_time='time', agg_method='attn',
                 attn_mode='prod', seq_len=None, n_head=4, drop_out=0.1, node_dim=None, time_dim=None):
        super(TGAN, self).__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.tune_times = 5
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.n_feat_th.shape[1]
        self.logger = logging.getLogger(__name__)

        self.n_feat_dim = self.n_feat_th.shape[1]
        self.model_dim = node_dim

        self.use_time = use_time
        self.batch_normal = torch.nn.BatchNorm1d(self.n_feat_dim, affine=False)
        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                                  self.feat_dim,
                                                                  self.feat_dim,
                                                                  attn_mode=attn_mode,
                                                                  n_head=n_head,
                                                                  drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:

            raise ValueError('invalid agg_method value, use attn or lstm')

        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        else:
            raise ValueError('invalid time option!')


        self._loss_bt = barlow_twins_loss
        self._loss_r = nn.MSELoss()
        self.R = 0  # radius R initialized with 0 by default.
        self.nu = 0.1
        self._loss_SVDD = Deep_SVDD


    # @torchsnooper.snoop()
    def forward(self, src_idx_l, target_idx_l, cut_time_l,  num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        link = torch.cat([src_embed, target_embed], dim=1)

        return link


    def get_radius(self, dist: torch.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

    def contrast(self, src_idx_l, target_idx_l,  cut_time_l, num_neighbors=20):  # background_idx_l,

        srca, targeta, at,srcb,  targetb, bt = augment_x(x=src_idx_l, y = target_idx_l,t= cut_time_l, p_x=0.8)
        srca_embed = self.tem_conv(srca, at, self.num_layers, num_neighbors)
        srcb_embed = self.tem_conv(srcb, bt, self.num_layers, num_neighbors)

        targeta_embed = self.tem_conv(targeta, at, self.num_layers, num_neighbors)
        targetb_embed = self.tem_conv(targetb, bt, self.num_layers, num_neighbors)
        linka = torch.cat([srca_embed, targeta_embed], dim=1)
        linkb = torch.cat([srcb_embed, targetb_embed], dim=1)

        score = torch.mean(linka,dim=1)
        loss = self._loss_bt(z_a=linka, z_b=linkb)
        return score, loss


    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert (curr_layers >= 0)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        src_node_t_embed = None

        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)

            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                src_idx_l,
                cut_time_l,
                num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)

            # get edge time features and node features
            src_ngh_t_embed = None

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]


            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   mask)
            return local

def augment_x(x: np, y: np, t: np, p_x: float):
    # device = x.device
    num_fts = x.shape[-1]

    a = bernoulli_mask(size=(1, num_fts), prob=p_x)
    x_a = (a * x).flatten()
    y_a = (a * y).flatten()
    a_t =(a * t).flatten()
    b = bernoulli_mask(size=(1, num_fts), prob=p_x)
    x_b = (b * x).flatten()
    y_b = (b * y).flatten()
    b_t = (b * t).flatten()

    return x_a,y_a, a_t, x_b,  y_b, b_t

def augment_g(edge_index: Tensor, p_e: float):
    device = edge_index.device
    ei = edge_index
    num_edges = ei.size(-1)

    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]
    ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]

    return ei_a, ei_b

def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return np.random.binomial(1, prob, size)
    # return torch.bernoulli((1 - prob) * torch.ones(size))

def _cross_correlation_matrix(
    z_a: Tensor, z_b: Tensor,
) -> Tensor:
    batch_size = z_a.size(0)

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    return c


def barlow_twins_loss(
    z_a: Tensor, z_b: Tensor,
) -> Tensor:
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Cross-correlation matrix
    c = _cross_correlation_matrix(z_a=z_a, z_b=z_b)

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * c[off_diagonal_mask].pow(2).sum()
    )

    return loss
def Deep_SVDD(output, R, c, nu, objective='hard_boundary'):
    dist = torch.sum((output-c) ** 2, dim=1)
    if objective == 'soft-boundary':
        scores = dist - R ** 2
        loss = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    else:
        scores = dist
        loss = torch.mean(dist)
    return scores,loss



