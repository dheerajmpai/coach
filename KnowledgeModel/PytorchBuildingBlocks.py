import torch
import numpy as np
import math
import torch

import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math


class LockedDropout(torch.nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or not self.prob:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.prob)
        mask = mask.div_(1 - self.prob)
        mask = mask.expand_as(x)
        return x * mask

class PermuteBatchSequenceLDBlock(torch.nn.Module):
    def __init__(self, locked_dropout):
      super().__init__()
      self.locked_dropout = locked_dropout

    def forward(self, x):
      x, lengths = pad_packed_sequence(x, batch_first = True)
      x = x.transpose(0, 1)
      if(self.locked_dropout is not None):
        x = self.locked_dropout(x)
      x = x.transpose(0, 1)
      x = pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted=False)
      return x

class PermuteBlock(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class pBLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()
        self.blstm = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1, bidirectional = True, batch_first = True, dropout = 0.3) 

    def forward(self, x_packed): 
        x_pad_packed, lengths = pad_packed_sequence(x_packed, batch_first = True)
        x_trunc, x_len = self.trunc_reshape(x_pad_packed, lengths)
        x_packed = pack_padded_sequence(x_trunc, x_len, batch_first = True, enforce_sorted=False)
        out1, (out2,out3) = self.blstm(x_packed)
        return out1

    def trunc_reshape(self, x, x_lens): 
        if(x.shape[1]%2 != 0):
          x = x[:,:-1,:]
        
        x_reshaped = x.reshape(x.shape[0], x.shape[1]//2, 2*x.shape[2])
        x_lens = x_lens//2
        return x_reshaped, x_lens
    
class Attention(torch.nn.Module):
  def __init__(self,listener_hidden_size,
              speller_hidden_size,
              projection_size):
    super().__init__()
    self.Wq = torch.nn.Linear(speller_hidden_size, projection_size, bias=False)
    self.Wk = torch.nn.Linear(listener_hidden_size, projection_size, bias=False)
    self.Wv = torch.nn.Linear(listener_hidden_size, projection_size, bias=False)
    self.projection_size = projection_size
    torch.nn.init.xavier_normal_(self.Wq.weight)
    torch.nn.init.xavier_normal_(self.Wk.weight)
    torch.nn.init.xavier_normal_(self.Wv.weight)
  
  def set_key_value(self, encoder_outputs):
    self.key = self.Wk(encoder_outputs) #(batch_size, timesteps, projection_size)
    self.value = self.Wv(encoder_outputs) #(batch_size, timesteps, projection_size)

  def compute_context(self, decoder_context):
    query = self.Wq(decoder_context) #(batch_size, projection_size)

    raw_weights = torch.einsum('Bp,Btp->Bt', query, self.key) / float(math.sqrt(self.projection_size))

    attention_weights = torch.nn.functional.softmax(raw_weights, dim = 1)

    attention_context = torch.einsum('Bt,Btp->Bp', attention_weights, self.value)

    return attention_context, attention_weights