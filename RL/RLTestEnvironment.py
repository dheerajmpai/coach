import numpy as np

import sys
sys.path.append("../SimulatedLearner")
sys.path.append("../Utils")

from BKTLearner import BKTLearner
from KMStateManager import KMStateManager
from RLEnvironment import RLEnvironment
from Utils  import load_data,convert_data_for_processing , get_word2vec
import wandb

class RLTestEnvironment(object):

  def __init__(self, env, exercise_list, exercise_embeddings, exercise_embedding_size, batch_size, epsilon, lr, gamma, num_episodes, max_iter):
    self.env = env
    self.exercise_embedding_size = exercise_embedding_size
    self.lr = lr
    self.gamma = gamma
    self.num_episodes = num_episodes
    self.exercise_list = exercise_list
    self.exercise_embeddings = exercise_embeddings
    self.batch_size = batch_size
    self.max_iter = max_iter

    self.weights = np.zeros((exercise_embedding_size +  len(env.get_state()) + 1) , dtype=float)
    self.num_batches = exercise_embeddings.shape[0]//batch_size
    
    self.epsilon = epsilon

  def getQValueForActions(self, actions):
    #Actions are embeddings
    current_state = self.env.get_state()
    q_values = []

    for i, action in enumerate(actions):
      concat_state = np.append(np.array([1]), np.concatenate((current_state,action)))
      q_val = np.sum(self.weights * concat_state)
      q_values.append(q_val)
    
    end_action_concat = np.append(np.array([1]), np.concatenate((current_state,np.zeros(self.exercise_embedding_size))))
    q_val = np.sum(self.weights * end_action_concat)
    q_values.append(q_val)

    return np.array(q_values)

  def run_one_episode(self):
    net_reward = 0;
    
    for batchIdx in range(self.num_batches):
      batch = self.exercise_list[batchIdx*self.batch_size:batchIdx*self.batch_size + self.batch_size]
      batch_embeddings = self.exercise_embeddings[batchIdx*self.batch_size:batchIdx*self.batch_size + self.batch_size]
      
      train_set = batch[0:int(0.9 * batch.shape[0])]
      test_set =  batch[int(0.9 * batch.shape[0]):]

      train_set_embeddings = batch_embeddings[0:int(0.9 * batch.shape[0])]
      test_set_embeddings = batch_embeddings[int(0.9 * batch.shape[0]):]

      self.env.start_episode(test_set)

      for iter in range(self.max_iter):
        choice = np.random.choice([0,1], p = [self.epsilon, 1 - self.epsilon])

        #Find the Qvalue for all actions. 
        q_value = []

        next_action = 0;
            
        if(choice == 0):
            #select action randomly
            next_action = np.random.choice(range(len(train_set_embeddings) + 1))
        else:
            #select the optimal action
            q_values = self.getQValueForActions(train_set_embeddings)
            next_action = np.argmax(q_values);
        

        episode_end = False
        if(next_action == train_set.shape[0]):
          episode_end = True
          next_state, reward = self.env.step(0, True)
          q_current_state = self.getQValueForActions(np.array([np.zeros(self.exercise_embedding_size)]))[0]
          current_state_with_bias = np.append(np.array([1]), np.concatenate((self.env.get_state(),np.zeros(self.exercise_embedding_size))))
        else:
          next_state, reward = self.env.step(train_set[next_action], (iter == (self.max_iter - 1)))
          q_current_state = self.getQValueForActions(np.array([train_set_embeddings[next_action]]))[0]
          current_state_with_bias = np.append(np.array([1]), np.concatenate((self.env.get_state(),train_set_embeddings[next_action])))
        if(reward != -1):
          net_reward += reward

        next_q_values = self.getQValueForActions(train_set_embeddings)
        max_q_next_state = np.max(next_q_values)

        #Update weights
        self.weights = self.weights - (self.lr * ((q_current_state - (reward + self.gamma * max_q_next_state)) * current_state_with_bias))

        if(episode_end):
          break

    return net_reward
  def run_training(self):
    reward_over_episodes = []
    for episode in range(self.num_episodes):
      self.env.reset()
      reward = self.run_one_episode()
      reward_over_episodes.append(reward)
      print(reward)    
    return reward_over_episodes


wandb.login(key="4a6e96eb645ce23f4ada4b7f5106dcbaed287c63")
run = wandb.init(
    name = "RL-Integrated-1-FullRun", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "ProjectFinalAblations" ### Project should be created in your wandb account 
)

training_data, training_labels = load_data("/content/drive/MyDrive/SLAM/data_en_es/en_es.slam.20190204.train")
test_data = load_data("/content/drive/MyDrive/SLAM/data_en_es/en_es.slam.20190204.dev")

exercices_merged, instanceIdExerciseMap,encodings, word_dict, pos_dict, format_dict, dependency_label_dict, morphological_feature_dict, wordVocab, posVocab, formatVocab, depLabelVocab, morphFeatureVocab = convert_data_for_processing(training_data, True, None, None, None, None, None)

#np.random.shuffle(all_ex)
learner = BKTLearner(len(word_dict),len(pos_dict), len(format_dict), len(dependency_label_dict),len(morphological_feature_dict), 0.1, 0.1, 0.001)
rlEnv = RLEnvironment(learner, wordVocab, posVocab, depLabelVocab, get_word2vec())
rlTestEnv = RLTestEnvironment(env = rlEnv, exercise_list = np.array(exercices_merged), exercise_embeddings = encodings, exercise_embedding_size = 768, batch_size = 50, epsilon=0.1, lr=0.1, gamma=0.1, num_episodes = 10, max_iter = 100)

rlTestEnv.run_training()