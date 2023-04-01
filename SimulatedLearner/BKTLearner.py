import numpy as np

class BKTLearner(object):
    
    def __init__(self, state_size, slip_prob, transition_prob, guess_prob):
        self.state_size = state_size
        self.state = np.zeros(self.state_size)
        self.slip_prob = slip_prob
        self.transition_prob = transition_prob
        self.guess_prob = guess_prob
    
    def reset(self):
        self.state = np.zeros(self.state_size)
        
    def predictAnswer(self, input):
        answer = []
        for token_index in input:
            p_correct = self.state[token_index] * (1 - self.slip_prob) + (1 - self.state[token_index]) * self.guess_prob 
            value = np.random.choice(np.array([0,1]), p = np.array([1 - p_correct, p_correct]))
            answer.append(value)
        return np.array(answer)
    
    def updateKnowledgeState(self, output_correctness, tokens):
        i = 0
        for token_index in tokens:
            if output_correctness[i] == 1:
                PLt_obs = self.state[token_index]*(1 - self.slip_prob) / (self.state[token_index]*(1 - self.slip_prob) + (1 - self.state[token_index])*self.guess_prob)
            else:
                PLt_obs = self.state[token_index]*(self.slip_prob) / (self.state[token_index]*(self.slip_prob) + (1 - self.state[token_index])*(1 - self.guess_prob))
            #print(PLt_obs ,self.state[token_index],self.slip_prob,self.guess_prob, output_correctness[i], token_index)
            self.state[token_index] = PLt_obs + (1 - PLt_obs)*self.transition_prob
            i += 1 
    
    def trainOneSet(self, excercises):
        for exercise in excercises:
            answer_correctness = self.predictAnswer(exercise)
            self.updateKnowledgeState(answer_correctness, exercise)
    
    def testOneSet(self, excercises):
        answer_correctness = []
        for exercise in excercises:
            answer_correctness_ex = self.predictAnswer(exercise)
            answer_correctness.append(answer_correctness_ex)
        return np.array(answer_correctness)
        
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
    
    def train(self, exercices_all, train_duration, test_duration):
        i = 0;
        accuracy = 0
        batch = 0
        
        while i < len(exercices_all):
            train_batch = exercices_all[i:train_duration + i]
            self.trainOneSet(train_batch)
            i += train_duration
            test_batch = exercices_all[i:i + test_duration]
            answer_correctness = self.testOneSet(test_batch)
            i += test_duration
            accuracy = self.computeAccuracyForTest(answer_correctness) 
            print("Batch + " + str(batch) + " " + " correct: " + str(accuracy))
            batch += 1
           
            
            
        
        