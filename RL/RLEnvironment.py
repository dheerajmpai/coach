import numpy as np
import wandb
from KMStateManager import KMStateManager

"""
This is the main environment that is used to train the RL Agent. It performs two functions:
1. Acts as the interface between the Agent and User/Simulated Learner and plays the actions
provides by the agent on the learner and observes the results (in the form of the correct/incorrect responses)
2. Builds an internal representation of the state of the world by making use of knowledge state
manager
"""
class RLEnvironment(object):

  def __init__(self, bkt_learner, bktTokenVocab,bktPosVocab,bktdepLabelVocab, word2Vec):
    self.bkt_learner = bkt_learner
    self.state_manager = KMStateManager(bktTokenVocab,bktPosVocab,bktdepLabelVocab,word2Vec)
    self.cummulative_score = 0
  
  def start_episode(self, test_set):
    self.test_set = test_set
    test_correctness = self.bkt_learner.testOneSetProbabilities(test_set)
    test_correctness = [np.where(test_correctness[i] >= 0.5 , 1, 0 ) for i in range(len(test_correctness))]
    self.start_score = self.bkt_learner.computeAccuracyForTest(test_correctness) #ToDo: Calculate start score on test set here.
    print("Start score: " + str(self.start_score))
 
  
  def reset(self):
    self.bkt_learner.reset()
    self.state_manager.reset()

  def get_state(self):
    return self.state_manager.get_state()

    # return np.concatenate((self.bkt_learner.token_state, 
    #                       self.bkt_learner.pos_state, 
    #                        self.bkt_learner.format_state ,
    #                       self.bkt_learner.dependency_state,
    #                       self.bkt_learner.morphological_state), axis = 0)
  
  def get_available_actions(self):
    return self.train_set_embeddings, self.train_set

  #This function is called by the RL agent to perform one step. 
  def step(self, action, isEpisodeEnd):
    if not isEpisodeEnd:
      answer_correctness = self.bkt_learner.trainOneExercise(action)
      state = self.state_manager.update_state(action, answer_correctness)

    if not isEpisodeEnd:
      return self.get_state(), -1
    else:
      test_correctness = self.bkt_learner.testOneSetProbabilities(self.test_set)
      test_correctness = [np.where(test_correctness[i] >= 0.5 , 1, 0 ) for i in range(len(test_correctness))]
      score = self.bkt_learner.computeAccuracyForTest(test_correctness)
      print("Score : " + str(score))
      self.cummulative_score += score - self.start_score
      wandb.log({"reward": score - self.start_score, "cummulative reward":self.cummulative_score})
      return self.get_state(), score - self.start_score #Calculate end score on test set here.

  