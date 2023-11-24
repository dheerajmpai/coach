import argparse
from collections import defaultdict, namedtuple
from io import open
import math
import os
from random import shuffle, uniform
from datetime import datetime
from future.utils import iterkeys, iteritems

from future.builtins import range
from future.utils import iteritems
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

from sentence_transformers import SentenceTransformer
sBertModel = SentenceTransformer('all-mpnet-base-v2')

def get_word2vec():
    #Prepare Glove Vectors : The file path is the first parameter
    word2vec = {}
    with open("/content/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for l in f:
            line = l.split()
            word = line[0]
            vect = np.array(line[1:]).astype(float)
            word2vec[word] = vect
    pickle.dump(word2vec, open(f'/content/drive/MyDrive/SLAM/6B.50_word2Vec.pkl', 'wb'))

    word2vec = pickle.load(open(f'/content/drive/MyDrive/SLAM/6B.50_word2Vec.pkl', 'rb'))

    return word2vec

#Helper Classes for data loading
class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance. This is meant to act as easy access to the
    data, and provides a launching point for deriving your own features from the data.
    """
    def __init__(self, instance_properties):

        # Parameters specific to this instance
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        # Derived parameters specific to this instance
        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])

        # Derived parameters specific to this exercise
        self.exercise_id = self.instance_id[:10]

        # Parameters shared across the whole session
        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']
        self.prompt = instance_properties.get('prompt', None)

        # Derived parameters shared across the whole session
        self.session_id = self.instance_id[:8]

    def to_features(self):
        """
        Prepares those features that we wish to use in the LogisticRegression example in this file. We introduce a bias,
        and take a few included features to use. Note that this dict restructures the corresponding features of the
        input dictionary, 'instance_properties'.

        Returns:
            to_return: a representation of the features we'll use for logistic regression in a dict. A key/feature is a
                key/value pair of the original 'instance_properties' dict, and we encode this feature as 1.0 for 'hot'.
        """
        to_return = dict()

        to_return['bias'] = 1.0
        to_return['user:' + self.user] = 1.0
        to_return['format:' + self.format] = 1.0
        to_return['token:' + self.token.lower()] = 1.0

        to_return['part_of_speech:' + self.part_of_speech] = 1.0
        for morphological_feature in self.morphological_features:
            to_return['morphological_feature:' + morphological_feature] = 1.0
        to_return['dependency_label:' + self.dependency_label] = 1.0
        
        time = datetime.now()
        if(time.second %10 == 0 and time.microsecond == 0):
          print(time)
          
        return to_return
    
    
def load_data(filename):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional): if you specified training data, a dict of instance_id:label pairs.
    """

    # 'data' stores a list of 'InstanceData's as values.
    data = []

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = False
    if filename.find('train') != -1:
        training = True

    if training:
        labels = dict()

    num_exercises = 0
    print('Loading instances...')
    instance_properties = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')
                instance_properties = dict()

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                if 'prompt' in line:
                    instance_properties['prompt'] = line.split(':')[1]
                else:
                    list_of_exercise_parameters = line[2:].split()
                    for exercise_parameter in list_of_exercise_parameters:
                        [key, value] = exercise_parameter.split(':')
                        if key == 'countries':
                            value = value.split('|')
                        elif key == 'days':
                            value = float(value)
                        elif key == 'time':
                            if value == 'null':
                                value = None
                            else:
                                assert '.' not in value
                                value = int(value)
                        instance_properties[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()
                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]

                instance_properties['token'] = line[1]
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]
                instance_properties['dependency_edge_head'] = int(line[5])
                if training:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label
                data.append(InstanceData(instance_properties=instance_properties))

        print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')

    if training:
        return data, labels
    else:
        return data
    
def load_labels(filename):
    """
    This loads labels, either the actual ones or your predictions.

    Parameters:
        filename: the filename pointing to your labels

    Returns:
        labels: a dict of instance_ids as keys and labels between 0 and 1 as values
    """
    labels = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                line = line.split()
            instance_id = line[0]
            label = float(line[1])
            labels[instance_id] = label
    return labels


def compute_acc(actual, predicted):
    """
    Computes the accuracy of your predictions, using 0.5 as a cutoff.

    Note that these inputs are lists, not dicts; they assume that actual and predicted are in the same order.

    Parameters (here and below):
        actual: a list of the actual labels
        predicted: a list of your predicted labels
    """
    num = len(actual)
    acc = 0.
    for i in range(num):
        if round(actual[i], 0) == round(predicted[i], 0):
            acc += 1.
    acc /= num
    return acc


def compute_avg_log_loss(actual, predicted):
    """
    Computes the average log loss of your predictions.
    """
    num = len(actual)
    loss = 0.

    for i in range(num):
        p = predicted[i] if actual[i] > .5 else 1. - predicted[i]
        loss -= math.log(p)
    loss /= num
    return loss


def compute_auroc(actual, predicted):
    """
    Computes the area under the receiver-operator characteristic curve.
    This code a rewriting of code by Ben Hamner, available here:
    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    num = len(actual)
    temp = sorted([[predicted[i], actual[i]] for i in range(num)], reverse=True)

    sorted_predicted = [row[0] for row in temp]
    sorted_actual = [row[1] for row in temp]

    sorted_posterior = sorted(zip(sorted_predicted, range(len(sorted_predicted))))
    r = [0 for k in sorted_predicted]
    cur_val = sorted_posterior[0][0]
    last_rank = 0
    for i in range(len(sorted_posterior)):
        if cur_val != sorted_posterior[i][0]:
            cur_val = sorted_posterior[i][0]
            for j in range(last_rank, i):
                r[sorted_posterior[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_posterior)-1:
            for j in range(last_rank, i+1):
                r[sorted_posterior[j][1]] = float(last_rank+i+2)/2.0

    num_positive = len([0 for x in sorted_actual if x == 1])
    num_negative = num - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if sorted_actual[i] == 1])
    auroc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive))

    return auroc


def compute_f1(actual, predicted, cutoff = 0.5):
    """
    Computes the F1 score of your predictions. Note that we use 0.5 as the cutoff here.
    """
    num = len(actual)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for i in range(num):
        if actual[i] >= cutoff and predicted[i] >= cutoff:
            true_positives += 1
        elif actual[i] < cutoff and predicted[i] >= cutoff:
            false_positives += 1
        elif actual[i] >= cutoff and predicted[i] < cutoff:
            false_negatives += 1
        else:
            true_negatives += 1

    try:
        precision = true_positives / (true_positives + false_positives)
        print(precision)
        recall = true_positives / (true_positives + false_negatives)
        print(recall)
        F1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        F1 = 0.0

    return F1


def evaluate_metrics(actual, predicted):
    """
    This computes and returns a dictionary of notable evaluation metrics for your predicted labels.
    """
    acc = compute_acc(actual, predicted)
    avg_log_loss = compute_avg_log_loss(actual, predicted)
    auroc = compute_auroc(actual, predicted)
    F1 = compute_f1(actual, predicted)

    return  acc, avg_log_loss,  auroc, F1


def test_metrics():
    actual = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
    predicted = [0.8, 0.2, 0.6, 0.3, 0.1, 0.2, 0.3, 0.9, 0.2, 0.7]
    metrics = evaluate_metrics(actual, predicted)
    metrics = {key: round(metrics[key], 3) for key in iterkeys(metrics)}
    assert metrics['accuracy'] == 0.700
    assert metrics['avglogloss'] == 0.613
    assert metrics['auroc'] == 0.740
    assert metrics['F1'] == 0.667
    print('Verified that our environment is calculating metrics correctly.')
    

def convert_data_for_processing(dataset, isTrain, word_dict, pos_dict, format_dict, dependency_label_dict, morphological_feature_dict):
  print(len(dataset))
  #Convert into data that can be used to train the BKT agent
  if isTrain:
    exercices = {}
    instanceIdExerciseMap = {}
    word_dict = {
        'unk': 0
    }
    wordVocab = []
    pos_dict = {
        'unk': 0
    }
    posVocab = []
    format_dict = {
        'unk': 0
    }
    formatVocab = []
    dependency_label_dict = {
        'unk': 0
    }
    depLabelVocab = []
    morphological_feature_dict = {
        'unk': 0
    }
    morphFeatureVocab = []

    unique_word_index = 1;
    unique_pos_index = 1;
    unique_format_index = 1;
    unique_dependency_label_index = 1;
    unique_morphological_feature_index = 1;

  for instance in dataset:
    user = instance.user
    instance_id = instance.instance_id[:-2]
    if user not in exercices:
      exercices[user] = {}
    if instance_id not in exercices[user] :
      exercices[user][instance_id] = []
    if instance_id not in instanceIdExerciseMap:
      instanceIdExerciseMap[instance_id] = []
    
    token = instance.token.lower()
    part_of_speech =  instance.part_of_speech.lower()
    format = instance.format.lower()
    dependency_label = instance.dependency_label.lower()
    morphological_features = instance.morphological_features

    token_info = []

    if token not in word_dict:
      if isTrain:
        word_dict[token] = unique_word_index
        wordVocab.append(token)
        unique_word_index += 1
      else:
        token = '<unk>'
    token_info.append(word_dict[token])

    instanceIdExerciseMap[instance_id].append(token)
    if part_of_speech not in pos_dict:
      if isTrain:
        pos_dict[part_of_speech] = unique_pos_index
        posVocab.append(part_of_speech)
        unique_pos_index += 1
      else:
        part_of_speech = 'unk'
    token_info.append(pos_dict[part_of_speech])

    if format not in format_dict:
      if isTrain:
        format_dict[format] = unique_format_index
        formatVocab.append(format)
        unique_format_index += 1
      else:
        format = 'unk'
    token_info.append(format_dict[format])

    if dependency_label not in dependency_label_dict:
      if isTrain:
        dependency_label_dict[dependency_label] = unique_dependency_label_index
        depLabelVocab.append(dependency_label)
        unique_dependency_label_index += 1
      else:
        dependency_label = 'unk'
    token_info.append(dependency_label_dict[dependency_label])


    morphology = []
    for feature_key, feature_val in morphological_features.items():
      key_val = str(feature_key) + ":" + str(feature_val)
      if key_val in morphological_feature_dict:
        morphology.append(morphological_feature_dict[key_val])
      else:
        if isTrain:
          morphological_feature_dict[key_val]= unique_morphological_feature_index
          morphFeatureVocab.append(key_val)
          unique_morphological_feature_index +=1 
        else:
          key_val = 'unk'
        morphology.append(morphological_feature_dict[key_val])

    token_info.append(morphology)
    exercices[user][instance_id].append(token_info)

  #Done Processing all data. Now final processing
  exercices_merged = []
  for user in exercices:
      for instance_id in exercices[user]:
          exercices_merged.append(exercices[user][instance_id])

  for instance_id in instanceIdExerciseMap:
      instanceIdExerciseMap[instance_id] = " ".join(instanceIdExerciseMap[instance_id])
  
  encodings = sBertModel.encode(list(instanceIdExerciseMap.values()), show_progress_bar = True)
  return exercices_merged, instanceIdExerciseMap,encodings, word_dict, pos_dict, format_dict, dependency_label_dict, morphological_feature_dict, wordVocab, posVocab, formatVocab, depLabelVocab, morphFeatureVocab