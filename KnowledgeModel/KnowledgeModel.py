"""
This file contains the main implementation required for the implementation of the 
knowledge model. The Knowledge model currently is implemented as an encoder-decoder
model. During training, the goal of the knowledge model is to make use of the 
history of the interactions of the user with the agent, condense that into a 
single vector ("knowledge state") and then make use of that knowledge state 
to predict the performance of the user on an unknown test set.  
"""

import torch
import numpy as np
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from PytorchBuildingBlocks import LockedDropout, PermuteBatchSequenceLDBlock, pBLSTM, Attention

"""
The encoder for the knowledge model.
"""
class Encoder(torch.nn.Module):
    def __init__(self, token_embedding_weight_matrix, encoder_hidden_size, token_vocabulary_size, pos_vocabulary_size, depLabelVocab_size, wordLabelVocab_size):
        super(Encoder, self).__init__()

        self.token_embedding = torch.nn.Embedding(token_vocabulary_size, 50)
        self.token_embedding.weight = torch.nn.Parameter(torch.from_numpy(token_embedding_weight_matrix))
        self.pos_embedding = torch.nn.Embedding(pos_vocabulary_size, 10)
        self.dependency_embedding = torch.nn.Embedding(depLabelVocab_size, 10)
        self.word_label_embedding = torch.nn.Embedding(wordLabelVocab_size, 15)

        self.locked_dropout = LockedDropout(0.5)
        self.lstm1 = torch.nn.LSTM(input_size = 86, hidden_size = encoder_hidden_size, num_layers = 1, bidirectional = True, batch_first = True, dropout = 0.3)
        self.lstm2 = torch.nn.LSTM(input_size = 2*encoder_hidden_size, hidden_size = encoder_hidden_size, num_layers = 4, bidirectional = True, batch_first = True, dropout = 0.5)
        self.pBLSTMs = torch.nn.Sequential( 
            PermuteBatchSequenceLDBlock(self.locked_dropout),
            pBLSTM(4*encoder_hidden_size, encoder_hidden_size),
            PermuteBatchSequenceLDBlock(self.locked_dropout),
            pBLSTM(4*encoder_hidden_size, encoder_hidden_size),
            PermuteBatchSequenceLDBlock(self.locked_dropout),
            pBLSTM(4*encoder_hidden_size, encoder_hidden_size)
        )
    def forward(self, x, x_lens, labels):
        token_embeddings = self.token_embedding(x[:,:,0].clone().to(torch.int64))
        pos_embeddings = self.pos_embedding(x[:,:,1].clone().to(torch.int64))
        dependency_embeddings = self.dependency_embedding(x[:,:,2].clone().to(torch.int64))
        word_labels_embeddings = self.word_label_embedding(x[:,:,3].clone().to(torch.int64))

        concatenated_out = torch.cat((
            token_embeddings.type(torch.float),
            pos_embeddings.type(torch.float),
            dependency_embeddings.type(torch.float), 
            labels.reshape(labels.shape[0], labels.shape[1],1).type(torch.float),
            word_labels_embeddings.type(torch.float)), dim=2)
        
        packed_out = pack_padded_sequence(concatenated_out, x_lens, batch_first = True, enforce_sorted=False)
        
        out = self.lstm1(packed_out)[0]
        out1 = self.lstm2(out)[0]#residual

        out_unpacked, out_unpacked_lens = pad_packed_sequence(out, batch_first = True)
        out1_unpacked, out1_unpacked_lens = pad_packed_sequence(out1, batch_first = True)
        out = out_unpacked + out1_unpacked
        out = pack_padded_sequence(out, out_unpacked_lens, batch_first = True, enforce_sorted=False)
        
        out = self.pBLSTMs(out)
        encoder_outputs, encoder_lens = pad_packed_sequence(out, batch_first = True)
        return encoder_outputs, encoder_lens

"""
The decoder for the knowledge model.
"""
class Decoder(torch.nn.Module):

    def __init__(self,attender: Attention,token_embedding_weight, decoder_hidden_size,projection_size,token_vocabulary_size,pos_vocabulary_size,depLabelVocab_size, output_size = 1):
        super().__init__()

        #self.user_embedding = torch.nn.Embedding(len(user_vocabulary), 10)
        self.token_embedding = torch.nn.Embedding(token_vocabulary_size, 50)
        self.token_embedding.weight = token_embedding_weight
        self.pos_embedding = torch.nn.Embedding(pos_vocabulary_size, 10)
        self.dependency_embedding = torch.nn.Embedding(depLabelVocab_size, 10)

        self.projection_size = projection_size
        self.attention = attender

        self.lstm1 = torch.nn.LSTM(input_size = 70, hidden_size = decoder_hidden_size, num_layers = 5, bidirectional = True, batch_first = True, dropout = 0.3)
        self.multi_head_attention1 = torch.nn.MultiheadAttention(2*decoder_hidden_size, 8, batch_first=True)
        self.batchnorm1 = torch.nn.BatchNorm1d(2*decoder_hidden_size)
        self.gelu = torch.nn.GELU()

        self.mlp_size = 2*decoder_hidden_size + projection_size
        self.mlp = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.mlp_size), 
            torch.nn.GELU(),
            torch.nn.Linear(self.mlp_size, self.mlp_size//8),
            torch.nn.BatchNorm1d(self.mlp_size//8), 
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.mlp_size//8, output_size),
        )
        

    def forward(self, x, x_lens, encoder_out):
        token_embeddings = self.token_embedding(x[:,:,0].clone().to(torch.int64))
        pos_embeddings = self.pos_embedding(x[:,:,1].clone().to(torch.int64))
        dependency_embeddings = self.dependency_embedding(x[:,:,2].clone().to(torch.int64))

        concatenated_out = torch.cat((
            token_embeddings.type(torch.float),
            pos_embeddings.type(torch.float),
            dependency_embeddings.type(torch.float)
            ), dim=2)

        packed_out = pack_padded_sequence(concatenated_out, x_lens, batch_first = True, enforce_sorted=False)

        out = self.lstm1(packed_out)[0]
        decoder_output, decoder_lens = pad_packed_sequence(out, batch_first = True)

        decoder_output = self.multi_head_attention1(decoder_output,decoder_output,decoder_output)[0] + decoder_output
        decoder_output = self.gelu(self.batchnorm1(decoder_output.permute(0,2,1)).permute(0,2,1))

        mlp_outputs = []
        
        for t in range(decoder_output.shape[1]):
          temp = self.mlp(torch.cat((decoder_output[:,t,:], encoder_out), dim = 1))
          mlp_outputs.append(temp)

        return torch.stack(mlp_outputs, dim = 1)

"""
Combined class for the knowledge model.
"""
class KnowledgeModel(torch.nn.Module):

    def __init__(self,token_vocabulary, pos_vocab_size, depLabelVocab_size, wordLabelVocab_size, word2vec, encoder_hidden_size=256,decoder_hidden_size=512,projection_size = 128, output_size=1):
        super().__init__()

        #Prepare GloVe for encoder
        matrix_len = len(token_vocabulary)
        weights_matrix = np.zeros((matrix_len, 50))
        words_found = 0

        for i, word in enumerate(token_vocabulary):
          try: 
              weights_matrix[i] = word2vec[word]
              words_found += 1
          except KeyError:
            if(i == 0):
              weights_matrix[i] = np.zeros((50, ))
            else:
              weights_matrix[i] = np.random.normal(scale=0.6, size=(50, ))

        self.encoder =  Encoder(weights_matrix, encoder_hidden_size, len(token_vocabulary), pos_vocab_size, depLabelVocab_size, wordLabelVocab_size) # TODO: Initialize Encoder
        self.attention = Attention(2*encoder_hidden_size,2*decoder_hidden_size,projection_size)

        self.decoder = Decoder(self.attention,self.encoder.token_embedding.weight, decoder_hidden_size,projection_size,len(token_vocabulary), pos_vocab_size, depLabelVocab_size, output_size) # TODO: Initialize Decoder 
        self.multi_head_attention1 = torch.nn.MultiheadAttention(2*encoder_hidden_size, 8, batch_first=True)
        self.batchnorm1 = torch.nn.BatchNorm1d(2*encoder_hidden_size)
        self.multi_head_attention2 = torch.nn.MultiheadAttention(2*encoder_hidden_size, 8, batch_first=True)
        self.batchnorm2 = torch.nn.BatchNorm1d(2*encoder_hidden_size)
        self.gelu = torch.nn.GELU()

    def forward(self, x_encoder,x_encoder_lengths, y_encoder_labels, x_decoder = None, x_decoder_lengths = None, return_knowledge_state = False):
        encoder_out, encoder_lens = self.encoder(x_encoder, x_encoder_lengths, y_encoder_labels)
        
        encoder_out = self.multi_head_attention1(encoder_out, encoder_out, encoder_out)[0] + encoder_out
        encoder_out = self.gelu(self.batchnorm1(encoder_out.permute(0,2,1)).permute(0,2,1))
        encoder_out = self.multi_head_attention2(encoder_out, encoder_out, encoder_out)[0] + encoder_out
        encoder_out = self.gelu(self.batchnorm2(encoder_out.permute(0,2,1)).permute(0,2,1))
        
        encoder_out_new = torch.zeros((encoder_out.shape[0], encoder_out.shape[2])).cuda()
        for batch_idx in range(encoder_out.shape[0]):
            encoder_out_new[batch_idx] = torch.max(encoder_out[batch_idx,0:encoder_lens[batch_idx]], dim = 0)[0]

        if return_knowledge_state:
            return encoder_out_new

        decoder_out  = self.decoder(x_decoder,  x_decoder_lengths, encoder_out_new)
        return decoder_out