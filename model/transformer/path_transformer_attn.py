
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from itertools import combinations
import copy


tgt_vocab = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4,
             'friend': 5, 'zero': 6, 'girl': 7, 'boy': 8, 'S': 9, 'E': 10, '.': 11}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 8  # enc_input max sequence length
tgt_len = 7  # dec_input(=dec_output) max sequence length



class MyDataSet(Data.Dataset):

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# loader = Data.DataLoader(
#     MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)


# ====================================================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k, seq_max_len):
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    
    pad_attn_mask = seq_k.data.eq(seq_max_len).unsqueeze(1)
    
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k, attn_matrix):
       
       
        attn_matrix = attn_matrix.to(torch.float32)
        scores = (torch.matmul(Q, K.transpose(-1, -2))  +  0.2 * attn_matrix.to(Q.device)) / np.sqrt(d_k)
     
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)  
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        # context：[[z1,z2,...],[...]]
        return context, attn


def get_index(path_lists, matrix):
    index = copy.deepcopy(path_lists)
    indices = []
    i = 0
    for path in index:
        path[0 : ] = range(i, i + len(path))
        i = i + len(path)

    for path_index in index:
        if len(path_index)==1:
            index_tuples = [(path_index[0], path_index[0])]
        else:
            index_tuples = list(combinations(path_index, 2))
        indices.extend(index_tuples)

    for i in range(len(indices)):
        matrix[indices[i][0:2]] = 1

    # matrix = torch.log(matrix)
    return matrix


def get_attn_matrix(paths_seq, paths_list):
    dimension = len(paths_seq)
    attn_matrix = np.zeros((dimension, dimension))
    attn_matrix = get_index(paths_list, attn_matrix)
    return attn_matrix


class MultiHeadAttention(nn.Module):
   

    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        self.d_v = params.d_v
        self.d_k = params.d_k
        self.n_heads = params.n_heads
        self.W_Q = nn.Linear(params.d_model, self.d_k * self.n_heads,
                             bias=False) 
        self.W_K = nn.Linear(params.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(params.d_model, self.d_v * self.n_heads, bias=False)
       
        self.fc = nn.Linear(self.n_heads * self.d_v, params.d_model, bias=False)
        self.d_model = params.d_model

    def forward(self, input_Q, input_K, input_V, attn_mask, attn_matrix):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
       
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q.to(input_Q.device)(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K.to(input_K.device)(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V.to(input_V.device)(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

      
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        attn_matrix = torch.from_numpy(attn_matrix)
        # attn_matrix = attn_matrix.add(-1)
        # attn_matrix = attn_matrix * 1e-9
        # attn_matrix_new = attn_matrix.repeat(batch_size, self.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, self.d_k, attn_matrix)
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.d_v)

        output = self.fc.to(context.device)(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(context.device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, params):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(params.d_model, params.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(params.d_ff, params.d_model, bias=False)
        )
        self.d_model = params.d_model
    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc.to(inputs.device)(inputs)
        # [batch_size, seq_len, d_model]
        d_model = self.d_model
        return nn.LayerNorm(d_model).to(output.device)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(params)
        self.pos_ffn = PoswiseFeedForwardNet(params)

    def forward(self, enc_inputs, enc_self_attn_mask, enc_matrix):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # enc_inputs * W_Q = Q
        # enc_inputs * W_K = K
        # enc_inputs * W_V = V
        # enc_inputs = enc_inputs.to(device)
        enc_self_attn_mask = enc_self_attn_mask.to(enc_inputs.device)
        enc_outputs, attn = self.enc_self_attn.to(enc_inputs.device)(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask, enc_matrix)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(params)
        self.dec_enc_attn = MultiHeadAttention(params)
        self.pos_ffn = PoswiseFeedForwardNet(params)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)  
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)  
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        # dec_self_attn, dec_enc_attn
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, rel_emb, params):
        super(Encoder, self).__init__()
        # rel_vocab_size = rel_emb.weight.size()
        self.src_emb = rel_emb  # src_vocab: rel_num
        # self.src_emb.weight.data[: rel_vocab_size[0]] = rel_emb.weight.data
        self.pos_emb = PositionalEncoding(
            params.d_model)  
        self.layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layers)])
        self.rel_num = int(rel_emb.weight.size()[0])

    def forward(self, enc_inputs, paths_seq, paths_list):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb.to(enc_inputs.device)(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb.to(enc_inputs.device)(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        
        attn_matrix = get_attn_matrix(paths_seq, paths_list)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.rel_num)  # [batch_size, src_len, src_len]
        enc_self_attns = [] 
        for layer in self.layers:  
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask, attn_matrix)  
            enc_self_attns.append(enc_self_attn) 
        # concatenate every layer
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(
            tgt_vocab_size, params.d_model) 
        self.pos_emb = PositionalEncoding(params.d_model)
        self.layers = nn.ModuleList([DecoderLayer(params)
                                     for _ in range(params.n_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   
        """
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(dec_inputs.device)  # [batch_size, tgt_len, d_model]
        
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(dec_inputs.device)  # [batch_size, tgt_len, tgt_len]
      
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(dec_inputs.device)  # [batch_size, tgt_len, tgt_len]

        
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).to(dec_self_attn_pad_mask.device)  
        
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
           
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns

def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        
        dec_input = torch.cat([dec_input.to(enc_input.device),
                               torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(enc_input.device)],
                              -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
       
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["E"]:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
    #     -1)
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict

class Transformer(nn.Module):
    def __init__(self, src_emb, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_emb.to(params.gpu), params).to(params.gpu)
        self.decoder = Decoder(params).to(params.gpu)
        # self.projection = nn.Linear(
        #     d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        """Transformers
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def path_emb_trans(rel_emb, path_inputs, params):
    src_emb = rel_emb # relation_embeddings 
    model = Transformer(src_emb, params).to(src_emb.device)
    enc_outputs, enc_self_attns =  model.encoder(path_inputs)

