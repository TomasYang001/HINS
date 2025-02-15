from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from sklearn.model_selection import KFold
import copy
import random
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(20191180)

def config():
    config = {}
    config['emb_size'] = 1300
    config['dropout_rate'] = 0.1

    # DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3

    # Encoder
    config['intermediate_size'] = 1536
    config['num_attention_heads'] = 4
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    return config

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

class CNN(nn.Module):
    def __init__(self, num_filters, k_size, hid_feats):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=72, out_channels=num_filters * 2, kernel_size=k_size, stride=1, padding=k_size // 2),
        )


        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size // 2),
        )

        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 8, k_size, 1, k_size // 2),
        )

        self.out = nn.AdaptiveAvgPool1d(1)

        self.layer1 = nn.Sequential(
            nn.Linear(hid_feats, hid_feats - 3 * (k_size - 1)),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hid_feats, hid_feats - 3 * (k_size - 1)),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = torch.FloatTensor(std.size()).normal_(0, 0.1)
        eps = Variable(eps).to(device)
        return eps.mul(std).add_(mean)


    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        # output = self.out(x)
        # output = output.squeeze()
        output1 = self.layer1(x)
        output1 = output1.to(device)
        output2 = self.layer2(x)
        output2 = output2.to(device)
        output = self.reparametrize(output1, output2)
        # output = output.permute(0, 2, 1)
        return output, output1, output2


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=2):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:

            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)

        return logits, att_maps


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=2):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
       # x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 3

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query).to(device)
        key = self.fc_k(key).to(device)
        value = self.fc_v(value).to(device)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))
        attn1 = attn1.to(device)

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn2 = attn2.to(device)
        attn = (attn1 + attn2) / self.scale


        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_v1 = r_v1.to(device)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        r_v2 = r_v2.to(device)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = weight2.to(device)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        x = x.to(device)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.MHA = MultiHeadAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob).to(device)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # context_layer = self.MHA(mixed_query_layer, mixed_key_layer, mixed_value_layer).to(device)
        # print(context_layer.shape)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #
        # # print("attention_scores:", attention_scores.shape)
        # attention_scores = attention_scores + attention_mask
        #
        # # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #
        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        #
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            # if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        # if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states

class TestBAN(nn.Module):
    def __init__(self, **config):
        super(TestBAN, self).__init__()

        self.bcn = weight_norm(
            BANLayer(v_dim=35, q_dim=17, h_dim=1300, h_out=1),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(1300, 1024, 64, binary=1)
        self.HLA_emb = nn.Linear(1300, 1280)
        self.Anti_emb = nn.Linear(1300, 1280)

        self.HLA_CNN = CNN(32,1,35)
        self.Anti_CNN = CNN(32,1,17)
        self.layers = 2
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']

        self.encoder = Encoder_MultipleLayers(self.layers, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.Layer1 = nn.Linear(6272,1024)

        self.Layer2 = nn.Linear(1024, 384)

        self.Layer3 = nn.Linear(384, 128)

        self.out = nn.Linear(128, 1)


    def forward(self, HLA, Antigen):

        # HLA = self.HLA_emb(HLA)
        #HLA = HLA.permute(0,2,1)
        #HLA_, HLA1, HLA2 = self.HLA_CNN(HLA)
        #HLA_.to(device)
        HLA = self.encoder(HLA)
        HLA = HLA.permute(0, 2, 1)
        HLA = HLA.to(device)
        # print("HLA_shape:", HLA_.shape)
        # print("HLA shpe:", HLA.shape)

        # Antigen = self.Anti_emb(Antigen)
        #Antigen = Antigen.permute(0,2,1)
        #Antigen_, Antigen1, Antigen2 = self.Anti_CNN(Antigen)
        #Antigen_ = Antigen_.to(device)

        Antigen = self.encoder(Antigen)
        Antigen = Antigen.permute(0, 2, 1)
        Antigen = Antigen.to(device)
        # print("Antigen shape:", Antigen.shape)

        f, att = self.bcn(HLA, Antigen)


        # features = torch.cat((HLA, Antigen), 2)
        # print(features.shape)
        # x = torch.reshape(features,(-1, 6272))
        # print(x.shape)

        # Input = torch.cat((HLA,Antigen),1)
        # print("input1:",len(Input))
        # print("f:", f.shape)
        out = self.mlp_classifier(f)
        # x = self.Layer1(x)
        # x = self.Layer2(x)
        # x = self.Layer3(x)
        # out = self.out(x)

        return out

def prepare_data(HLA, Antigen, label):
    dataset = [[]]
    for ind in range(len(HLA)):
        hla = HLA[ind]
        dataset[ind].append(np.array(hla, dtype=np.float32))
        antigen = Antigen[ind]
        dataset[ind].append(np.array(antigen, dtype=np.float32))
        Y = label[ind]
        dataset[ind].append(np.array(Y, dtype=np.float32))
        if ind < len(HLA) - 1:
            dataset.append([])
    return dataset


label = np.load('NAgdata/Test/Rizvi_B_train_label.npy')

Antigen_B = np.load('NAgdata/Test/Rizvi_B_train_antigen.npy')
HLA_B = np.load('NAgdata/Test/Rizvi_B_train_HLA.npy')
HLA_B = torch.FloatTensor(HLA_B)
Antigen_B = torch.FloatTensor(Antigen_B)

Antigen_E = torch.load('NAgdata/Test/Rizvi_ESM2_Antigen.pt')
HLA_E = torch.load('NAgdata/Test/Rizvi_ESM2_HLA.pt')
HLA_E = torch.FloatTensor(HLA_E)
Antigen_E = torch.FloatTensor(Antigen_E)

HLA = torch.cat((HLA_B, HLA_E), 2)
Antigen = torch.cat((Antigen_B, Antigen_E), 2)

HLA_E = None
HLA_B = None
Antigen_B = None
Antigen_E = None

HLA = torch.FloatTensor(HLA)
Antigen = torch.FloatTensor(Antigen)

test_Dataset = prepare_data(HLA, Antigen, label)
test_loader = DataLoader(dataset=test_Dataset, batch_size=128, shuffle = True)
test_loss = 0
y_label, y_pred = [], []

Cf = config()
print(device)
model = TestBAN(**Cf).to(device)

model.load_state_dict(torch.load('DA3.pth'))


model.eval()

with torch.no_grad():
    model.eval()
    for testhla, testantigen, testY in tqdm(test_loader):
        thla = torch.Tensor(testhla).to(device)
        tantigen = torch.Tensor(testantigen).to(device)
        labels = torch.Tensor(testY).to(device)

        tscore = model(thla, tantigen).to(device)
        tn, tloss = binary_cross_entropy(tscore, labels)

        test_loss += tloss.item()
        y_label = y_label + labels.to(device).tolist()
        y_pred = y_pred + tn.to(device).tolist()
test_loss = test_loss / 64
print("test Loss:{}".format(test_loss))
auroc = roc_auc_score(y_label, y_pred)
auprc = average_precision_score(y_label, y_pred)

fpr, tpr, thresholds = roc_curve(y_label, y_pred)
prec, recall, _ = precision_recall_curve(y_label, y_pred)
precision = tpr / (tpr + fpr)
f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
thred_optim = thresholds[5:][np.argmax(f1[5:])]
y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
cm1 = confusion_matrix(y_label, y_pred_s)
accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
# sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
# specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

print('Accuracuy:{}, ROC:{}, PR:{}'.format(accuracy, auroc, auprc))
