from dgl_bfs import _bfs_relational
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def findAllPath(graph, start, end, hop, rel=None, path=[], ent_path=[]):
    if len(path) == 0:
        path = path + [[rel, start]]
    else:
        path = path + [[path[-1][-1], rel, start]]
    ent_path = ent_path + [start]
    if start == end:
        return [path], [ent_path]

    paths = []  
    ent_paths = []
    for rel in range(2):
        node_lists = graph.get((start, rel))
        if len(graph.get((start, rel))) > 10:
            node_lists = random.sample(node_lists, 10)
        if node_lists is not None:
            if hop == 0:
                break
            for node in node_lists:
                if node not in ent_path:
                    newpaths, newentpaths = findAllPath(graph, node, end, hop - 1, rel, path, ent_path)
                    for newpath in newpaths:
                        paths.append(newpath)
                    for newentpath in newentpaths:
                        ent_paths.append(newentpath)
    return paths, ent_paths

def findAllPath_before(graph, start, end, hop, rel=None, path=[]):
    if len(path) == 0:
        path = path + [[rel, start]]
    else:
        path = path + [[path[-1][-1], rel, start]]
    if start == end:
        return [path]

    paths = []  
    for rel in range(2):
        if graph.get((start, rel)) is not None:
            if hop == 0:
                break
            for node in graph.get((start, rel)):
                if node not in path:
                    newpaths = findAllPath_before(graph, node, end, hop-1, rel, path)
                    for newpath in newpaths:
                        paths.append(newpath)
    return paths

def change_paths(path):
    del(path[0])
    return path

def get_triple_path(paths, matrix):
    rel_paths = list()
    for path in paths:
        rel_path = list()
        for i in range(len(path) - 1):
            path_rel_label = []
            for adj in range(len(matrix)):
                # rel_ = adj
                exist_rel = matrix[adj][path[i], path[i + 1]]
                if exist_rel > 0:
                    path_rel_label.append(adj)
                    pre_path = rel_path
            rel_path = pre_path + [list(path_rel_label)]
        rel_paths.append(rel_path)
    return rel_paths

def find_paths(roots, end, adj, h, max_nodes_per_hop=None, path=[]):
    roots=set([roots])
    path = path + list(roots)
    if roots == set([end]):
        return [path]
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    nodeList = next(bfs_generator)
    paths = list()
    for node in nodeList:
        try:
            # nodeList=next(bfs_generator)
            if node not in path:
                if h == 0:
                    break
                else:
                    newpaths = find_paths(node, end, adj, h - 1, max_nodes_per_hop, path)

                    for newpath in newpaths:
                        paths.append(newpath)
        except StopIteration:
            pass
    return paths

# def change_paths(paths):
#     for path in paths:
#
# '''edgeLinks{
#   "1":["2" "5"]
#   "2":["1" "3" "4"]
#   "3":["2" "4" "5"]
#   "4":["2" "3" "5"]
#   "5":["1" "3" "4"]
#   }'''
# graph = dict({(1, 0): [2, 5],
#               (2, 0): [2, 3, 4, 5],
#               (3, 1): [4, 5],
#               (4, 1): [5],
#               (5, 0): [6, 7]})
#
# # graph_matrix = [[1,0,2], [1,0,5], [2,0,2], [2,0,3], [2,0,4], [2,0,5], [3,1,4], [3,1,5], [4,1,5], [5,0,6], [5,0,7]]
#
# row_1 = np.array([1,1,2,2,2,2,3,3,4,5,5]) # 行索引
# col_1 = np.array([2,5,2,3,4,5,4,5,5,6,7]) # 列索引
# data_1 = np.array([1,1,1,1,1,1,1,1,1,1,1]) # 索引对应的数值
# rel = [0,0,0,0,0,0,1,1,1,0,0]
# matrix = ssp.coo_matrix((data_1, (row_1, col_1)), shape=(8, 8)).toarray()
#
# paths_before = findAllPath_before(graph, 2, 5, 2, path=[])
# paths, ent_paths = findAllPath(graph, 2, 5, 2, path=[], ent_path=[])
#
# paths_triples_before = list(map(change_paths, paths_before))
# paths_triples = list(map(change_paths, paths))

import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)

        output = self.out(concat)
        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output



class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_feedforward, dropout):
        super().__init__()

        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Multi-head attention
        src2 = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(src2))

        # Feedforward layer
        src2 = self.feedforward(src)
        src = self.norm2(src + self.dropout(src2))

        return src


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_head, n_encoder_layers, d_feedforward, dropout, src_pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, d_feedforward, dropout) for _ in range(n_encoder_layers)])
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, src):
        # src = [src_len, batch_size]
        batch_size = src.shape[1]
        src_len = src.shape[0]
        pos = torch.arange(0, src_len).unsqueeze(1).repeat(1, batch_size).to(device)
        # pos = [src_len, batch_size]
        # src = torch.tensor(src).to(torch.int64)
        # src = self.embedding(src)
        # src = src * self.scale
        src = self.pos_encoding(src)
        src_mask = make_src_mask(src, self.src_pad_idx)
        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [src_len, batch_size, d_model]
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_feedforward, dropout):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoder_attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        # Multi-head attention on decoder inputs
        tgt2, _ = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        tgt = tgt + self.dropout(self.norm1(tgt2))

        # Multi-head attention on encoder outputs
        tgt2, _ = self.encoder_attention(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_padding_mask)
        tgt = tgt + self.dropout(self.norm2(tgt2))

        # Feedforward layer
        tgt2 = self.feedforward(self.norm3(tgt))
        tgt = tgt + self.dropout(tgt2)

        return tgt


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_head, n_decoder_layers, d_feedforward, dropout, tgt_pad_idx):
        super().__init__()

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, d_feedforward, dropout) for _ in range(n_decoder_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.tgt_pad_idx = tgt_pad_idx

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # tgt: [batch_size, tgt_seq_len]
        # memory: [batch_size, src_seq_len, d_model]
        # tgt_mask: [tgt_seq_len, tgt_seq_len]
        # memory_mask: [tgt_seq_len, src_seq_len]

        tgt_seq_len, batch_size = tgt.size()
        tgt_positions = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(1).expand(tgt_seq_len, batch_size)

        tgt = self.dropout(self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model) + tgt_positions))

        for decoder_layer in self.decoder_layers:
            tgt, attention_weights = decoder_layer(tgt, memory, tgt_mask, memory_mask)

        logits = self.fc_out(tgt)
        return logits


class TransformerPS(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 d_model,
                 n_head,
                 n_encoder_layers,
                 n_decoder_layers,
                 d_feedforward,
                 dropout,
                 src_pad_idx,
                 trg_pad_idx):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head, n_encoder_layers, d_feedforward, dropout, src_pad_idx)
        self.decoder = Decoder(trg_vocab_size, d_model, n_head, n_decoder_layers, d_feedforward, dropout, trg_pad_idx)
        self.out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src, trg):
        src_mask = make_src_mask(src, self.src_pad_idx)
        trg_mask = make_trg_mask(trg, self.trg_pad_idx)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        output = self.out(output)
        return output, attention


def make_src_mask(src, src_pad_idx):
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def make_trg_mask(trg, trg_pad_idx):
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

d_model = 512
n_head = 8
n_encoder_layers = 6
n_decoder_layers = 6
d_feedforward = 2048
dropout = 0.1
src_vocab_size = 10000
tgt_vocab_size = 10000
src_pad_idx = 0
tgt_pad_idx = 0
device = 'cuda'

transformerps = TransformerPS(src_vocab_size,
                            tgt_vocab_size,
                            d_model,
                            n_head,
                            n_encoder_layers,
                            n_decoder_layers,
                            d_feedforward,
                            dropout,
                            src_pad_idx,
                            tgt_pad_idx)

# Generate multiple sequence vectors
seq1 = torch.randn(10, d_model)
seq2 = torch.randn(10, d_model)
seq3 = torch.randn(10, d_model)

# Aggregate the sequence vectors using the Transformer
seq_list = [seq1, seq2, seq3]
seq_tensor = torch.stack(seq_list)
output = transformerps.encoder(seq_tensor)

# The output is a tensor of shape (batch_size, seq_len, d_model), where batch_size = len(seq_list)
print(output.shape)
