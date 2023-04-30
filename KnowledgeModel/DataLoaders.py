import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import sys
sys.path.append("../Utils")

from Utils.Utils import load_data, load_labels

user_keyed_data = {}
user_keyed_label = {}

user_idx = 1
user2Idx = {"unk" : 0}
idx2User = {0: "unk"}
user_vocabulary = ["unk"]

token_idx = 1
word2Idx = {"unk" : 0}
token_vocabulary = ["unk"]


pos_idx = 1
pos2Idx = {"unk" : 0}
pos_vocabulary = ["unk"]

morph_idx = 1
morph2Idx = {"unk" : 0}
morph_vocab = ["unk"]

dep_label_idx = 1
depLabel2Idx = {"unk": 0}
depLabelVocab = ["unk"]

word_label_idx = 1
wordLabel2Idx = {"unk": 0}
wordLabelVocab = ["unk"]

min_days = float('inf')
max_days = -float('inf')
max_time = -float('inf')
min_time = float('inf')

user_keyed_data = {}
user_keyed_label = {}

def prepare_data():
    global user_idx
    global user2Idx
    global user_vocabulary
    global idx2User

    global token_idx
    global word2Idx
    global token_vocabulary

    global pos_idx
    global pos2Idx
    global pos_vocabulary

    global morph_idx
    global morph2Idx
    global morph_vocab

    global dep_label_idx
    global depLabel2Idx
    global depLabelVocab

    global word_label_idx
    global wordLabel2Idx
    global wordLabelVocab
    
    global min_days
    global max_days
    global max_time
    global min_time
    
    global user_keyed_data
    global user_keyed_label
    
    user_keyed_data = {}
    user_keyed_label = {}

    user_idx = 1
    user2Idx = {"unk" : 0}
    idx2User = {0: "unk"}
    user_vocabulary = ["unk"]

    token_idx = 1
    word2Idx = {"unk" : 0}
    token_vocabulary = ["unk"]


    pos_idx = 1
    pos2Idx = {"unk" : 0}
    pos_vocabulary = ["unk"]

    morph_idx = 1
    morph2Idx = {"unk" : 0}
    morph_vocab = ["unk"]

    dep_label_idx = 1
    depLabel2Idx = {"unk": 0}
    depLabelVocab = ["unk"]

    word_label_idx = 1
    wordLabel2Idx = {"unk": 0}
    wordLabelVocab = ["unk"]
    
    #Add path to files here
    training_data, training_labels = load_data("/content/en_es/en_es.slam.20190204.train")
    
    valid_data = load_data("/content/en_es/en_es.slam.20190204.dev")
    valid_labels = load_labels("/content/en_es/en_es.slam.20190204.dev.key")
    
    test_data = load_data("/content/en_es/en_es.slam.20190204.test")
    test_labels = load_labels("/content/en_es/en_es.slam.20190204.test.key")
    
    
    for instance in training_data:
        days = instance.days
        time = instance.time
        if(time is None or time < 0):
            time = 0

        if(instance.days > max_days):
            max_days = instance.days
        if(instance.days < min_days):
            min_days = instance.days
        if(time > max_time):
            max_time = time
        if(time < min_time):
            min_time = time
    
    train_data = ExcerciseDataset(training_data, training_labels, 256)
    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data, 
        num_workers = 8,
        batch_size  = 64, 
        pin_memory  = True,
        shuffle     = True,
        collate_fn = ExcerciseDataset.collate_fn
    )
    val_dataset = ExcerciseValidationDataset(valid_data, valid_labels, 128)
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_dataset, 
        num_workers = 8,
        batch_size  = 64, 
        pin_memory  = True,
        shuffle     = False,
        collate_fn = ExcerciseValidationDataset.collate_fn
    )
    test_dataset = ExcerciseValidationDataset(test_data, test_labels, 128)
    test_loader = torch.utils.data.DataLoader(
        dataset     = test_dataset, 
        num_workers = 8,
        batch_size  = 64, 
        pin_memory  = True,
        shuffle     = False,
        collate_fn = ExcerciseValidationDataset.collate_fn
    )
    # sanity check
    i = 0;
    for data in train_loader:
        x_encoder,x_decoder, y_encoder,y_decoder, lx_encoder,lx_decoder, ly_encoder, ly_decoder = data
        print(x_encoder.shape,x_decoder.shape, y_encoder.shape,y_decoder.shape, lx_encoder.shape, lx_decoder.shape,ly_encoder.shape, ly_decoder.shape)
        print(y_decoder[1])
        tokens = x_decoder[0, :, 1]
        string_tok = []
        for token in tokens:
            string_tok.append(token_vocabulary[int(token)])
        print(" ".join(string_tok))
        tokens = x_encoder[0, :, 1]
        string_tok = []
        for token in tokens:
            string_tok.append(token_vocabulary[int(token)])
        print(" ".join(string_tok))
        i += 1
        if(i==2):
            break
    return train_loader, val_loader, test_loader, training_labels, valid_labels, test_labels
     
class ExcerciseDataset(torch.utils.data.Dataset):
    
    def get_relevant_history(self, candidate, query,sequence_length):
      candidate = torch.stack(candidate, dim = 0)
      score = torch.zeros(candidate.shape[0])
      for tok in query:
        score += torch.sum((candidate == tok).type(torch.int), dim = 1)

      ind = torch.sort(torch.argsort(score)[-sequence_length:])[0]
      history = []
      
      for idx in ind:
        history.append(candidate[idx, :])

      del candidate, score
      return history, ind

    def __init__(self, data, labels, sequence_size): 
        global user_idx
        global user2Idx
        global user_vocabulary

        global token_idx
        global word2Idx
        global token_vocabulary

        global pos_idx
        global pos2Idx
        global pos_vocabulary

        global morph_idx
        global morph2Idx
        global morph_vocab

        global dep_label_idx
        global depLabel2Idx
        global depLabelVocab

        global word_label_idx
        global wordLabel2Idx
        global wordLabelVocab
        
        if len(user_keyed_data) == 0:
          for i, instance in enumerate(data):
            user = instance.user
            if user not in user_keyed_data:
              user_keyed_data[user] = []
              user_keyed_label[user] = []
            
            exercise = []
    
            token = instance.token.lower()
            pos_tag = instance.part_of_speech.lower()
            dependency_label = instance.dependency_label.lower()
            time = instance.time
            if time is None or time < 0 or time > 100:
              time = 0
            days = instance.days
            if days is None or days > 100:
              days = 0
            days = float((days - min_days)/float(max_days - min_days))
            time = float((time - min_time)/float(max_time - min_time))
            if(days > 1):
              days = 1
            if(time > 1):
              time = 1

            label = labels[instance.instance_id]
            
            if user not in user2Idx:
              user2Idx[user] = user_idx
              idx2User[user_idx] = user
              user_vocabulary.append(user)
              user_idx += 1

            exercise.append(user2Idx[user])

            if token not in word2Idx:
              word2Idx[token] = token_idx
              token_vocabulary.append(token)
              token_idx += 1

            exercise.append(word2Idx[token])

            if pos_tag not in pos2Idx:
              pos2Idx[pos_tag] = pos_idx
              pos_vocabulary.append(pos_tag)
              pos_idx += 1

            exercise.append(pos2Idx[pos_tag])

            if dependency_label not in depLabel2Idx:
              depLabel2Idx[dependency_label] = dep_label_idx
              depLabelVocab.append(dependency_label)
              dep_label_idx += 1

            exercise.append(depLabel2Idx[dependency_label])
            exercise.append(float(days))
            exercise.append(float(time))

            all_combined1 = str(token) + str(0)
            all_combined2 = str(token) + str(1)

            if all_combined1 not in wordLabel2Idx:
              wordLabel2Idx[all_combined1] = word_label_idx
              wordLabelVocab.append(all_combined1)
              word_label_idx += 1

            if all_combined2 not in wordLabel2Idx:
              wordLabel2Idx[all_combined2] = word_label_idx
              wordLabelVocab.append(all_combined2)
              word_label_idx += 1
            
            if(label == 0):
              exercise.append(wordLabel2Idx[all_combined1])
            else:
              exercise.append(wordLabel2Idx[all_combined2])
            
            user_keyed_data[user].append(torch.tensor(exercise, dtype = float))
            user_keyed_label[user].append(label)
          

        #At this point we have the exercises for each user.

        self.timesteped_data = []
        self.timesteped_labels = []
        batch_bar = tqdm(total=len(user_keyed_data), dynamic_ncols=True, leave=False, position=0, desc='Train')

        for user in user_keyed_data:
          for i in range(0, len(user_keyed_data[user]), 30):
            chunk = user_keyed_data[user][i:i + sequence_size]
            if(len(chunk) < 128):
              continue;
            self.timesteped_data.append(torch.stack(chunk,dim=0))
            self.timesteped_labels.append(torch.tensor(user_keyed_label[user][i:i + sequence_size]))
          
          batch_bar.update()

        self.length = len(self.timesteped_data)
        
    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        data = self.timesteped_data[ind] # TODO
        labels = self.timesteped_labels[ind] # TODO
        return data, labels


    def collate_fn(batch):
        batch_data_encoder = [x[0:len(x)//2] for x,y in batch] # TODO
        batch_data_decoder = [x[len(x)//2:] for x,y in batch] # TODO
        batch_labels_encoder = [y[0:len(y)//2] for x,y in batch] # TODO
        batch_labels_decoder = [y[len(y)//2:] for x,y in batch] 

        batch_data_encoder_pad = pad_sequence(batch_data_encoder, batch_first=True, padding_value=0) # TODO
        batch_data_decoder_pad = pad_sequence(batch_data_decoder, batch_first=True, padding_value=0)

        encoder_lengths_data = [len(x) for x in batch_data_encoder] # TODO
        decoder_lengths_data = [len(x) for x in batch_data_decoder] 

        batch_labels_encoder_pad = pad_sequence(batch_labels_encoder, batch_first=True, padding_value=2) # TODO
        batch_labels_decoder_pad = pad_sequence(batch_labels_decoder, batch_first=True, padding_value=2)
        encoder_lengths_labels =  [len(x) for x in batch_labels_encoder] # TODO
        decoder_lengths_labels =  [len(x) for x in batch_labels_decoder] # TODO
        
        return batch_data_encoder_pad,batch_data_decoder_pad, batch_labels_encoder_pad,batch_labels_decoder_pad, torch.tensor(encoder_lengths_data), torch.tensor(decoder_lengths_data), torch.tensor(encoder_lengths_labels), torch.tensor(decoder_lengths_labels)
    

class ExcerciseValidationDataset(torch.utils.data.Dataset):
    def get_relevant_history(self, candidate, query,sequence_length):
      candidate = torch.stack(candidate, dim = 0)
      score = torch.zeros(candidate.shape[0])
      for tok in query:
        score += torch.sum((candidate == tok).type(torch.int), dim = 1)

      ind = torch.sort(torch.argsort(score)[-sequence_length:])[0]
      history = []
      
      for idx in ind:
        history.append(candidate[idx, :])

      del candidate, score
      return history, ind

    def __init__(self, data, labels, sequence_size=128): 

        self.user_keyed_valid_data = {}
        self.user_keyed_valid_label = {}
        self.sequence_size = sequence_size

        global user_idx
        global user2Idx
        global user_vocabulary

        global token_idx
        global word2Idx
        global token_vocabulary

        global pos_idx
        global pos2Idx
        global pos_vocabulary

        global morph_idx
        global morph2Idx
        global morph_vocab

        global dep_label_idx
        global depLabel2Idx
        global depLabelVocab

        global word_label_idx
        global wordLabel2Idx
        global wordLabelVocab

        
        for i, instance in enumerate(data):
          user = instance.user
          if user not in self.user_keyed_valid_data:
            self.user_keyed_valid_data[user] = []
            self.user_keyed_valid_label[user] = []
          
          exercise = []
          
          token = instance.token.lower()
          pos_tag = instance.part_of_speech.lower()

          dependency_label = instance.dependency_label.lower()
          time = instance.time
          if time is None:
            time = 0
          days = instance.days
          if days is None:
            days = 0
          days = float((days - min_days)/(max_days - min_days))
          time = float((time - min_time)/(max_time - min_time))
          if(days > 1):
            print("days maxed")
            days = 1
          if(time > 1):
            print("time maxed")
            time = 1

          label = labels[instance.instance_id]

          assert user in user2Idx
          exercise.append(user2Idx[user])

          if token not in word2Idx:
            token = "unk"

          exercise.append(word2Idx[token])

          if pos_tag not in pos2Idx:
            pos_tag = "unk"

          exercise.append(pos2Idx[pos_tag])

          if dependency_label not in depLabel2Idx:
            dependency_label = "unk"

          exercise.append(depLabel2Idx[dependency_label])
          exercise.append(float(days))
          exercise.append(float(time))

          all_combined = str(token) + str(label)
          if all_combined not in wordLabel2Idx:
            all_combined = "unk"
          exercise.append(wordLabel2Idx[all_combined])

          self.user_keyed_valid_data[user].append(torch.tensor(exercise, dtype = float))
          self.user_keyed_valid_label[user].append(label)
          

        #At this point we have the exercises for each user.

        self.timesteped_data = []
        self.timesteped_labels = []

        batch_bar = tqdm(total=len(self.user_keyed_valid_data), dynamic_ncols=True, leave=False, position=0, desc='Train') 

        for user in self.user_keyed_valid_data:
          for i in range(0, len(self.user_keyed_valid_data[user]), sequence_size):
            chunk1 = user_keyed_data[user][-sequence_size:] #History for the user
            assert len(chunk1) > 0
            chunk2 = self.user_keyed_valid_data[user][i:i + sequence_size]
            assert len(chunk2) > 0
            chunk1.extend(chunk2)

            self.timesteped_data.append(torch.stack(chunk1,dim=0))
            self.timesteped_labels.append(torch.cat(
                (torch.FloatTensor(user_keyed_label[user][-sequence_size:]),
                torch.FloatTensor(self.user_keyed_valid_label[user][i:i + sequence_size])), 
                dim = 0
                ))
          batch_bar.update()
            
        
        self.length = len(self.timesteped_data)
        
    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        data = self.timesteped_data[ind] # TODO
        labels = self.timesteped_labels[ind] # TODO
        return data, labels


    def collate_fn(batch):
        
        batch_data_encoder = [] # TODO
        batch_data_decoder = [] # TODO

        batch_labels_encoder = [] # TODO
        batch_labels_decoder = [] 

        for x,y in batch:
          user_id =  idx2User[int(x[0][0].item())]
          sequence_size = 128
          if(len(user_keyed_data[user_id]) < sequence_size):
            sequence_size = len(user_keyed_data[user_id])

          batch_data_encoder.append(x[0:sequence_size])
          assert len(x[0:sequence_size]) > 0
          batch_data_decoder.append(x[sequence_size:])
          assert len(x[sequence_size:]) > 0
          batch_labels_encoder.append(y[0:sequence_size])
          assert len(y[0:sequence_size]) > 0
          batch_labels_decoder.append(y[sequence_size:])
          assert len(y[sequence_size:]) > 0

        batch_data_encoder_pad = pad_sequence(batch_data_encoder, batch_first=True, padding_value=0) # TODO
        batch_data_decoder_pad = pad_sequence(batch_data_decoder, batch_first=True, padding_value=0)

        encoder_lengths_data = [len(x) for x in batch_data_encoder] # TODO
        decoder_lengths_data = [len(x) for x in batch_data_decoder] 

        batch_labels_encoder_pad = pad_sequence(batch_labels_encoder, batch_first=True, padding_value=2) # TODO
        batch_labels_decoder_pad = pad_sequence(batch_labels_decoder, batch_first=True, padding_value=2)
        encoder_lengths_labels =  [len(x) for x in batch_labels_encoder] # TODO
        decoder_lengths_labels =  [len(x) for x in batch_labels_decoder] # TODO

        return batch_data_encoder_pad,batch_data_decoder_pad, batch_labels_encoder_pad,batch_labels_decoder_pad, torch.tensor(encoder_lengths_data), torch.tensor(decoder_lengths_data), torch.tensor(encoder_lengths_labels), torch.tensor(decoder_lengths_labels)