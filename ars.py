# AI 2018 
# @copyright Hadelin de Pontaves & Alexis Jacq 
# find the tutorial on https://www.superdatascience.com/

# Importing libraries 
import os 
import numpy as np

# Setting the Hyper Parameters : fixed value during the training 

# Through a class and an object of the class 
# Use of self -variable of the object of the class 
class Hp(): 
    def __init__(self):
        self.nb_steps = 1000 # training loops 
        self.episode_lenght = 1000 # maximum lenght of an episode
        self.learning_rate =  0.02 # how fast the AI is learning? 
        self.nb_directions = 16 # number of pertubations applied to the weights
        self.nb_best_directions = 16 # best directions we are going to keep. optimization of the model. 
        assert self.nb_best_directions <= self.nb_directions # self.nb_best_directions always inferior to self.nb_directions
        self.noise = 0.03 # sample the pertubation following a gaussian distribution : sigma. 
        self.seed = 1  # fixed the parameters of the environment / observe the same thing 
        self.env_name = '' # environment name 
        
        
# Normalizing the states 

# States are some vectors describing exactly what's happening at a time t. 
# Substract by the mean and divide by the standard deviation 
# Neural Network make better dissociation of values between 0 and 1 rather than 100 and 200. 
class Normalizer(): 
    def __init__(self,nb_inputs): # number of inputs of the perceptron = number of elements inside the vector describing the state.
        self.n = np.zeros(nb_inputs) # creation of a vector of nb_inputs initialize to zeros
        self.mean = np.zeros(nb_inputs) # initialize mean 
        self.mean_diff = np.zeros(nb_inputs) # initialize the mean-diff
        self.var = np.zeros(nb_inputs) # initialize the variance
        
    def observe(self, x):  # x is the new state. 
        self.n +=1.  # observe a new state 
        last_mean = self.mean.copy()
        self.mean += (x-self.mean)/self.n # compute the new mean. (in x-self.mean, self.mean is the last_mean)
        self.mean_diff += (x-last_mean)*(x-self.mean) # self.mean is the new mean. 
        self.var = (self.mean_diff/self.n).clip(min = 1e-2) # variance never equal to zero .clip(min = 1e-2)
        
    def normalize(self,inputs):
        obs_mean = self.mean  # observe mean when we reach a new state
        obs_std = np.sqrt(self.var) # observe variance when we reach a new state
        return (inputs-obs_mean)/obs_std 
        
    
# Building the AI
        
# self AI; 
# input_size = number of elements inside the vector describing the state
# output_size = number of actions to play (the AI is returning several action)
# metrics of weights of the perceptron is Theta
class Policy(): 
    def __init__(self, input_size, output_size): 
        self.theta = np.zeros((output_size,input_size)) # matrix multiplication by the left size

# delta is the small number following a normal distribution. 
# output without pertubation delta = None 
# three values for the direction : positive, negative, None. 
# evaluate can do three things : 2 'if' + 1 'else' = 3  
# 1/return an output when you feed a certain output 
# 2/ apply input +  positive perturbation => output 
# 3/ apply input +  opposite pertubation => Output 
    def evaluate(self,input,delta = None, direction = None): 
        if direction is None: 
            return self.theta.dot(input) # perceptron code applying no pertubation 
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input) #hp will be the object created from the class Hp
        else: 
            return (self.theta - hp.noise*delta).dot(input)
        
# create a function that sample delta following the normal distribution 
# follow a gaussian distribution , mean = 0, variance = 1 
# np.random.randn = n is for normal distribution 
# *self.theta.shape : matrix of small values same dimension of the matrix of weights theta.
# 16 matrices of small values nb_directions => 16 postive + 16 negative directions
# [] list of 16 matrices : for _ in range(hp.nb_directions)]             
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)] 


# Methode of infinite differences      
# Approximation of the gradient descent 
# Aim : increase the reward 
# Differentiation of the reward regarding to the weight
# rollouts => list of triplet 
#1/reward obtain in the positive direction, 
#2/reward obtain in the negative direction.
#3/the pertubation that give the two rewards 
    def update (self,rollouts,sigma_r): # sigma_r standard deviation of the reward
        step = np.zeros(self.theta.shape) # the update step 
        for r_pos,r_neg,d in rollouts :
            step +=(r_pos-r_neg)*d # the for loop allow to compute the sum
            self.theta += hp.learning_rate/(hp.nb_best_directions*sigma_r)*step #update the matrix of weights ##@            
            
# Exploring the policy on one specific direction and over one episode 
# One episode is made of few actions. 
def explore(env,normalizer, policy, direction = None, delta= None):
    state=env.reset()
    done = False # done = True if we reach the end of the episod 
    num_plays = 0. # number of actions played / float computation 
    sum_rewards = 0. # accumulated reward 
    while not done and num_plays <hp.episode_lenght: # episode lenght is the number of actions played in an episode
        normalizer.observe(state) # mean and the variance 
        state = normalizer.normalize(state)
    # not reach the end of the episode 
    # number of actions < episode 
    #=> cumulated reward
        action = policy.evaluate(state,delta,direction)
        state,reward,done,_ = env.step(action)
        reward = max(min(reward,1),-1) # force all the positive rewards = 1 and all the negative reward = -1  
        sum_rewards +=reward # cumulated reward
        num_plays +=1 # increment num_plays of one
    return sum_rewards # return the cumulated reward. 

# Training AI => Maximizing the reward.
# watch the videos recorded in exp > brs > monitor


    
    


        
    
    
    
