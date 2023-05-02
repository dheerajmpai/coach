# An Interactive Conversational Agent to Aid Human Learning

This repository consists of the implementation of a Conversational agent that can interact with users to help them learn a new language. This code repository currently consists of the following components:

1. The Knowledge Tracing Module: This will be responsible for making use of the user interaction history and build a representation of the knowledge state of the user. The relevant code regarding the same can be found under the "KnowledgeModel" directory. The main model is located in "KnowledgeModel.py" and the code used for training the model is located in "KnowledgeModelTraining.py". Other files contain auxiliary helper functions for data preparation etc. 

2. The Reinforcement Learning Agent: This is a Q-learning based agent that is currently trained to schedule exercises from a fixed pool for the user. The relevant code for the same is located under the "RL" folder. The "RLTestEnvironment.py" consists of the code for the RL agent and makes use of Q-learning. The "RLEnvironment.py" and "KMStateManager.py" are used to create create an environment to train the RL agent. 

3. BKTLearner.py : This is a simulated learner based on Bayesian Knowledge Tracing that is used in the absence of real users to train the RL agent.

Apart from the cleaned code that is present in the ".py" files, you can also find the code that was used for experimentation in the "notebooks" folder under each of the above-mentioned parent folders. 