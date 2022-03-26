import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .fc_encoder import FcEncoder

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                            bidirectional=True, num_layers=1)
    
    def forward(self, x, states=None):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        states: (h_0, c_0)
        '''
        r_out, (h_n, h_c) = self.rnn(x, states)
        return r_out, (h_n, h_c)

class AttentiveLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentiveLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                            bidirectional=True, num_layers=1)
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.se = nn.Sequential(
                nn.Conv1d(hidden_size*2, hidden_size // 2, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size // 2, hidden_size // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size // 2, hidden_size*2, kernel_size=1),
                nn.Sigmoid()
        )
        
        self.out_cnn = nn.Sequential(
                nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
        )
    
    def forward(self, x, states=None):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        states: (h_0, c_0)
        '''
        r_out, (h_n, h_c) = self.rnn(x, states)
        # attn = self.se(r_out.transpose(1, 2))
        # attn = attn.transpose(1, 2)
        # return r_out * attn, (h_n, h_c)
        return r_out, (h_n, h_c)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                       num_layers=1)
    
    def forward(self, x, states):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        states: (h_0, c_0)
        '''
        r_out, (h_n, h_c) = self.rnn(x, states)
        return r_out, (h_n, h_c)

class BiLSTM_official_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                            bidirectional=True, num_layers=1)
    
    def forward(self, x):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        return r_out, (h_n, h_c)

class LSTM_official_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_official_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                       num_layers=1)
    
    def forward(self, x):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        return r_out, (h_n, h_c)

class FcLstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(FcLstmEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = FcEncoder(input_size, [hidden_size, hidden_size], dropout=0.1, dropout_input=False)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True,
                       num_layers=1, bidirectional=bidirectional)
    
    def forward(self, x, states):
        x = self.fc(x)
        r_out, (h_n, h_c) = self.rnn(x, states)
        return r_out, (h_n, h_c)

class AttentionFusionNet(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, hidden_size):
        super(AttentionFusionNet, self).__init__()
        self.a_dim = a_dim
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.hidden_size = hidden_size
        self.mapping = nn.Linear(self.hidden_size, self.hidden_size)
        self.modality_context = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.modality_context.data.normal_(0, 0.05)
        self.A_conv = nn.Conv1d(a_dim, hidden_size, kernel_size=1, padding=0)
        self.V_conv = nn.Conv1d(v_dim, hidden_size, kernel_size=1, padding=0)
        self.L_conv = nn.Conv1d(l_dim, hidden_size, kernel_size=1, padding=0)
        self.rnn = self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, )
    
    def atten_embd(self, a_input, v_input, l_input):
        a_input = a_input.unsqueeze(-2) # [batch_size, seq_len, 1, embd_dim]
        v_input = v_input.unsqueeze(-2)
        l_input = l_input.unsqueeze(-2)
        data = torch.cat([a_input, v_input, l_input], dim=-2) # [batch_size, seq_len, 3, embd_dim]
        batch_size, seq_len, _, embd_dim = data.size()
        proj_data = torch.tanh(self.mapping(data))   # [batch_size, seq_len, 3, hidden_size]
        weight = F.softmax(data @ self.modality_context, dim=-2) # [batch_size, seq_len, 3, 1]
        fusion = torch.sum(data * weight, dim=-2)
        return fusion

    def forward(self, a_input, v_input, l_input, states):
        '''
        Input size [batch_size, seq_len, embd_dim]
        '''
        a_input = self.A_conv(a_input.transpose(1, 2)).permute(0, 2, 1)
        v_input = self.V_conv(v_input.transpose(1, 2)).permute(0, 2, 1)
        l_input = self.L_conv(l_input.transpose(1, 2)).permute(0, 2, 1)
        fusion = self.atten_embd(a_input, v_input, l_input) # [batch_size, seq_len, embd_dim]
        r_out, (h_n, h_c) = self.rnn(fusion, states)
        return r_out, (h_n, h_c)

class AttentionFusionNet2(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, hidden_size):
        super(AttentionFusionNet2, self).__init__()
        self.a_dim = a_dim
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.hidden_size = hidden_size
        self.mapping = nn.Linear(self.hidden_size, self.hidden_size)
        self.A_conv = nn.Conv1d(a_dim, hidden_size, kernel_size=1, padding=0)
        self.V_conv = nn.Conv1d(v_dim, hidden_size, kernel_size=1, padding=0)
        self.L_conv = nn.Conv1d(l_dim, hidden_size, kernel_size=1, padding=0)
        self.context_proj = nn.Linear(3 * hidden_size, hidden_size)
        self.rnn = self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, )
    
    def atten_embd(self, a_input, v_input, l_input):
        batch_size, seq_len, embd_dim = a_input.size()
        context = torch.cat([a_input, v_input, l_input], dim=-1)
        context = torch.tanh(self.context_proj(context)).view(-1, self.hidden_size, 1) # [batch_size * seq_len, hidden_size, 1]
        _a_input = a_input.contiguous().view(batch_size * seq_len, 1, self.hidden_size) # [batch_size * seq_len, 1, hidden_size]
        _v_input = v_input.contiguous().view(batch_size * seq_len, 1, self.hidden_size) # [batch_size * seq_len, 1, hidden_size]
        _l_input = l_input.contiguous().view(batch_size * seq_len, 1, self.hidden_size) # [batch_size * seq_len, 1, hidden_size]
        a_weight = torch.bmm(_a_input, context).view(batch_size, -1, 1)          # [batch_size, seq_len, 1]
        v_weight = torch.bmm(_v_input, context).view(batch_size, -1, 1)
        l_weight = torch.bmm(_l_input, context).view(batch_size, -1, 1)
        weight = torch.cat([a_weight, v_weight, l_weight], dim=-1) # [batch_size, seq_len, 3]
        weight = F.softmax(weight, dim=-1).unsqueeze(-1)
        data = torch.cat([a_input.unsqueeze(-2), v_input.unsqueeze(-2), l_input.unsqueeze(-2)], dim=-2)
        fusion = torch.sum(data * weight, dim=-2)
        return fusion

    def forward(self, a_input, v_input, l_input, states):
        '''
        Input size [batch_size, seq_len, embd_dim]
        '''
        a_input = self.A_conv(a_input.transpose(1, 2)).permute(0, 2, 1)
        v_input = self.V_conv(v_input.transpose(1, 2)).permute(0, 2, 1)
        l_input = self.L_conv(l_input.transpose(1, 2)).permute(0, 2, 1)
        fusion = self.atten_embd(a_input, v_input, l_input) # [batch_size, seq_len, embd_dim]
        r_out, (h_n, h_c) = self.rnn(fusion, states)
        return r_out, (h_n, h_c)


"""
class BiLSTMEncoder(nn.Module):
    ''' LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_size):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_size = embd_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.embd_size),
            nn.ReLU(),
        )

    def forward(self, x, length):
        batch_size = x.size(0)
        # x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        r_out, (h_n, h_c) = self.rnn(x)
        h_n = h_n.contiguous().view(batch_size, -1)
        embd = self.fc(h_n)
        return embd

class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='last'):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        assert embd_method in ['maxpool', 'attention', 'last']
        self.embd_method = embd_method

        if self.embd_method == 'maxpool':
            self.maxpool = nn.MaxPool1d(self.hidden_size)
        
        elif self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        ''''
        参考这篇博客的实现:
        https://blog.csdn.net/dendi_hust/article/details/94435919
        https://blog.csdn.net/fkyyly/article/details/82501126
        论文：Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        embd = self.maxpool(r_out.transpose(1,2))   # r_out.size()=>[batch_size, seq_len, hidden_size]
                                                    # r_out.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        return embd.squeeze()

    def embd_last(self, r_out, h_n):
        #Just for  one layer and single direction
        return h_n.squeeze()

    def forward(self, x):
        '''
        r_out shape: seq_len, batch, num_directions * hidden_size
        hn and hc shape: num_layers * num_directions, batch, hidden_size
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_'+self.embd_method)(r_out, h_n)
        return embd
"""

if __name__ == '__main__':
    # model = AttentionFusionNet2(100, 200, 300, 128)
    # a_input = torch.rand(12, 30, 100)
    # v_input = torch.rand(12, 30, 200)
    # l_input = torch.rand(12, 30, 300)
    # state = (torch.zeros(1, 12, 128), torch.zeros(1, 12, 128))
    # r_out, (h_n, h_c) = model(a_input, v_input, l_input, state)
    # print(r_out.shape)

    model = AttentiveLSTMEncoder(345, 256)
    input = torch.rand(32, 300, 345)
    out, _ = model(input)
    print(out.shape)