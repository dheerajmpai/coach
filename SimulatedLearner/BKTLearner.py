"""
This is the main class that helps in creating a simulated learner
which will be used to train the RL agent. 

The implementation is based on http://act-r.psy.cmu.edu/wordpress/wp-content/uploads/2012/12/893CorbettAnderson1995.pdf

Author: Deigant Yadava (dyadava)
"""
import numpy as np
import wandb

class BKTLearner(object):
    
    #Initialise the state of the BKT learner. This consists of the information of each skill being in the learnt state
    def __init__(self, token_state_size, pos_state_size, format_state_size, dependency_state_size, morphological_state_size, slip_prob, transition_prob, guess_prob):
        self.token_state_size = token_state_size
        self.pos_state_size = pos_state_size
        self.format_state_size = format_state_size
        self.dependency_state_size = dependency_state_size
        self.morphological_state_size = morphological_state_size

        self.token_state = np.full(self.token_state_size,0, dtype = np.float32)
        self.pos_state = np.full(self.pos_state_size,0,dtype = np.float32)
        self.format_state = np.full(self.format_state_size,0,dtype = np.float32)
        self.dependency_state = np.full(self.dependency_state_size,0,dtype = np.float32)
        self.morphological_state = np.full(self.morphological_state_size,0,dtype = np.float32)


        self.slip_prob = slip_prob
        self.transition_prob = transition_prob
        self.guess_prob = guess_prob
    
    #reset to initial state
    def reset(self):
        self.token_state = np.full(self.token_state_size,0, dtype = np.float32)
        self.pos_state = np.full(self.pos_state_size,0,dtype = np.float32)
        self.format_state = np.full(self.format_state_size,0,dtype = np.float32)
        self.dependency_state = np.full(self.dependency_state_size,0,dtype = np.float32)
        self.morphological_state = np.full(self.morphological_state_size,0,dtype = np.float32)
        

    #Get the probability of a token being in the learnt state
    def getNetLearnedProb(self,token_info):
      token_index = token_info[0]
      pos_index = token_info[1]
      format_index = token_info[2]
      dep_index = token_info[3]
      morphological_indices = token_info[4]
      net_learned = self.token_state[token_index] * self.pos_state[pos_index] #* self.format_state[format_index] * self.dependency_state[dep_index]
      for index in morphological_indices:
       net_learned *= self.morphological_state[index]

      return net_learned

    #Predict the probability of the user getting the answer correct
    def predictAnswerProbabilities(self, input):
        answer = []
        for token_info in input:
            net_learned = self.getNetLearnedProb(token_info)
            p_correct = net_learned * (1 - self.slip_prob) + (1 - net_learned) * self.guess_prob 
            answer.append(p_correct)
        return np.array(answer)

    #Predict a probabilistic answer of the user on the given exercise
    def predictAnswer(self, input):
        answer = []
        for token_info in input:
            net_learned = self.getNetLearnedProb(token_info)
            p_correct = net_learned * (1 - self.slip_prob) + (1 - net_learned) * self.guess_prob 
            value = np.random.choice(np.array([0,1]), p = np.array([1 - p_correct, p_correct]))
            answer.append(value)
        return np.array(answer)
    
    #Get the posterior probability after getting the observation
    def getPosterior(self, prob, output_correctness):
      if output_correctness == 1:
        posterior = prob*(1 - self.slip_prob) / (prob*(1 - self.slip_prob) + (1 - prob)*self.guess_prob)
      else:
        posterior = prob*(self.slip_prob) / (prob*(self.slip_prob) + (1 - prob)*(1 - self.guess_prob))
      return posterior

    #Update the state vector of the user based on the posterior probability
    def updateKnowledgeState(self, output_correctness, input):
        i = 0
        for token_info in input:
          #print("Updating token state for : " + str(token_info[0]))
          token_posterior = self.getPosterior(self.token_state[token_info[0]], output_correctness[i])
          self.token_state[token_info[0]] = token_posterior + (1 - token_posterior) * self.transition_prob          

          pos_posterior = self.getPosterior(self.pos_state[token_info[1]], output_correctness[i])
          self.pos_state[token_info[1]] = pos_posterior + (1 - pos_posterior) * self.transition_prob

          format_posterior = self.getPosterior(self.format_state[token_info[2]], output_correctness[i])
          self.format_state[token_info[2]] = format_posterior + (1 - format_posterior) *  self.transition_prob

          dep_posterior = self.getPosterior(self.dependency_state[token_info[3]], output_correctness[i])
          self.dependency_state[token_info[3]] = dep_posterior + (1 - dep_posterior) * self.transition_prob

          for index in token_info[4]:
           morpho_posterior = self.getPosterior(self.morphological_state[index], output_correctness[i])
           self.morphological_state[index] = morpho_posterior + (1 - morpho_posterior) * self.transition_prob
          i += 1 

    #'Train' the simulated learner on a set of exercises. This function predicts the 
    # response of the user on the exercises and then uses that to update the knowledge
    # state
    def trainOneSet(self, excercises):
        for exercise in excercises:
            answer_correctness = self.predictAnswer(exercise)
            self.updateKnowledgeState(answer_correctness, exercise)
    
    #Same as trainOneSet but for a single exercise.
    def trainOneExercise(self, exercise):
      answer_correctness = self.predictAnswer(exercise)
      self.updateKnowledgeState(answer_correctness, exercise)
      return answer_correctness

    #Generates the probability that the user will answer a question correctly
    #given a particular set of test exercises.
    def testOneSetProbabilities(self, excercises):
        answer_correctness = []
        for exercise in excercises:
            answer_correctness_ex = self.predictAnswerProbabilities(exercise)
            answer_correctness.append(answer_correctness_ex)
        return np.array(answer_correctness)

    #Predict the answers of the user on a set of test exercises
    def testOneSet(self, excercises):
        answer_correctness = []
        for exercise in excercises:
            answer_correctness_ex = self.predictAnswer(exercise)
            answer_correctness.append(answer_correctness_ex)
        return np.array(answer_correctness)
        
    #Helper function to compute accuracy.
    def computeAccuracyForTest(self, test_response):
        correct = 0;
        total = 0;
        for exercise in test_response:
            for token in exercise:
                correct += token
                total += 1
        
        if(total == 0):
          return 0

        return float(correct)/total * 100
    
    #Test function to run a simulation on a BKT learning.
    def train(self, exercices_all, train_duration, test_duration):
        i = 0;
        accuracy = 0
        batch = 0
        cummulative_reward = 0;
        
        while i < len(exercices_all):
            if(train_duration + i < len(exercices_all)):
              train_batch = exercices_all[i:train_duration + i]
            else: 
              train_batch = exercices_all[i:]
            i += train_duration
            if(i + test_duration < len(exercices_all)):
              test_batch = exercices_all[i:i + test_duration]
            else:
               test_batch = exercices_all[i:]
            i += test_duration

            answer_correctness_before = self.testOneSetProbabilities(test_batch)
            #print(answer_correctness_before)
            answer_correctness_before = [np.where(answer_correctness_before[i] >= 0.5 , 1, 0 ) for i in range(len(answer_correctness_before))]
            self.trainOneSet(train_batch)
            answer_correctness = self.testOneSetProbabilities(test_batch)
            answer_correctness = [np.where(answer_correctness[i] >= 0.5 , 1, 0 ) for i in range(len(answer_correctness))]

            accuracy_before = self.computeAccuracyForTest(answer_correctness_before) 
            accuracy = self.computeAccuracyForTest(answer_correctness) 
            print("Batch + " + str(batch) + " " + " correct before: " + str(accuracy_before) + " correct after: " + str(accuracy))
            cummulative_reward += accuracy - accuracy_before
            wandb.log({'Batch Accuracy After': accuracy, 'Batch Accuracy Before': accuracy_before, "reward": accuracy - accuracy_before, "cummulative reward":cummulative_reward})
            batch += 1