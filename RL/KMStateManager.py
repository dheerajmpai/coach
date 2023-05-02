import pickle
import torch

import sys
sys.path.append("../KnowledgeModel")
from KnowledgeModel import KnowledgeModel

"""
This class is a helper for the RL Environment. It keeps track of the history of interactions
of the agent with the user and then makes use of that history to generate the knowledge
state of the user. This class makes use of the pre-trained knowledge model to generate
the knowledge vector. The knowledge vector is used as the state vector in the RL Environment. 
"""
class KMStateManager(object):
  def load_pickle_obj(self, filename):
    with open(filename, "rb") as file:
      return pickle.load(file)

  def reset(self):
    self.most_recent_history = torch.zeros((1,128,4))
    self.most_recent_history_labels = torch.full((1,128), 2)
    self.most_recent_history_len = 0;
    self.current_state = torch.zeros((256))


  #Initialise the Knowledge state manager. Load the knowledge model weights along
  # with the token dictionary for the same.
  def __init__(self,bktTokenVocab,bktPosVocab,bktdepLabelVocab, word2VecDict):
    self.most_recent_history = torch.zeros((1,128,4))
    self.most_recent_history_labels = torch.full((1,128), 2)
    self.most_recent_history_len = 0
    self.current_state = torch.zeros((256))

    #Add path to the dictionaries for the pre-trained KM Model
    LSTM_TOKEN_DICT_FILE_NAME = "/content/drive/MyDrive/KM_word2Idx"
    LSTM_POS_DICT_FILE_NAME = "/content/drive/MyDrive/KM_pos2Idx"
    LSTM_DEP_LABEL_DICT_FILE_NAME  = "/content/drive/MyDrive/KM_depLabel2Idx"
    LSTM_WORD_LABEL_DICT_FILE_NAME = "/content/drive/MyDrive/KM_wordLabel2Idx"

    self.tokenDict = self.load_pickle_obj(LSTM_TOKEN_DICT_FILE_NAME)
    self.posDict = self.load_pickle_obj(LSTM_POS_DICT_FILE_NAME)
    self.depLabelDict = self.load_pickle_obj(LSTM_DEP_LABEL_DICT_FILE_NAME)
    self.wordLabelDict = self.load_pickle_obj(LSTM_WORD_LABEL_DICT_FILE_NAME)

    self.bktTokenVocab = bktTokenVocab
    self.bktPosVocab = bktPosVocab
    self.bktdepLabelVocab = bktdepLabelVocab

    lstm_model = KnowledgeModel(
        token_vocabulary = list(self.tokenDict.keys()),
        pos_vocab_size = len(list(self.posDict.keys())),
        depLabelVocab_size = len(list(self.depLabelDict.keys())),
        wordLabelVocab_size = len(list(self.wordLabelDict.keys())),
        word2vec = word2VecDict,
        encoder_hidden_size  = 128,
        decoder_hidden_size = 128,
        projection_size = 256,
        output_size = 3
      )
    
    kmModelState = torch.load("/content/drive/MyDrive/KM-ModelFinal3.pt")
    lstm_model.load_state_dict(kmModelState['model_state_dict'])

    lstm_model = lstm_model.cuda() #Push to GPU
    lstm_model.eval()
    self.encoder_model = lstm_model

  def get_state(self):
    return self.current_state
  
  #Update the history of the agent and make use of the updated history
  #to recompute the knowledge state.This should be called at every step 
  #the agent interacts with the user.  
  def update_state(self, exercise, answers):
    
    for i, token_info in enumerate(exercise):
      token = self.bktTokenVocab[token_info[0]]
      pos_tag = self.bktPosVocab[token_info[1]] 
      dependency_label = self.bktdepLabelVocab[token_info[3]]
      word_label_combined = str(token) + str(answers[i])

      word_exercise = torch.tensor([[
                                self.tokenDict[token], 
                                self.posDict[pos_tag], 
                                self.depLabelDict[dependency_label],
                                self.wordLabelDict[word_label_combined]
                                ]])
      label = torch.tensor([answers[i]])
      
      if(self.most_recent_history_len == 128):
        #history is full
        self.most_recent_history = torch.cat((self.most_recent_history, word_exercise.unsqueeze(0)), dim = 1)
        self.most_recent_history = self.most_recent_history[:,1:,:]

        self.most_recent_history_labels = torch.cat((self.most_recent_history_labels, label.unsqueeze(0)), dim = 1)
        self.most_recent_history_labels = self.most_recent_history_labels[:,1:]
      else:
        #history has space
        self.most_recent_history[0][self.most_recent_history_len] = word_exercise
        self.most_recent_history_labels[0][self.most_recent_history_len] = label
        
        self.most_recent_history_len += 1
    
    assert self.most_recent_history_len <= 128 and self.most_recent_history_len > 0
    assert self.most_recent_history.shape[0] == 1
    assert self.most_recent_history.shape[1] == 128
    assert self.most_recent_history.shape[2] == 4

    if(self.most_recent_history_len >= 12):
      self.current_state = self.encoder_model(self.most_recent_history.cuda(),[self.most_recent_history_len], self.most_recent_history_labels.cuda())[0]
      self.current_state = self.current_state.cpu().detach().numpy()

    return self.current_state