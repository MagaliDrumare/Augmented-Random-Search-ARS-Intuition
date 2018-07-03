# AI 2018 

# Importing libraries 
import os 
import numpy as np
import gym
from gym import wrappers
import pybullet_envs #HalfCheetahBulletEnv-v0 belong to pybullet_envs

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
        self.env_name = 'HalfCheetahBulletEnv-v0' # environment name 
        
        
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
        sum_rewards +=reward
        num_plays +=1  
    return sum_rewards

# Training AI 
def train(env, policy, normalizer,hp): # try other values for normalizer and hp
    for step in range (hp.nb_steps): 

    # Initializing the perturbations delta and the positive/ negative rewards
        deltas = policy.sample_deltas() # Get 16 matrices of perturbations. 
        positive_rewards =[0] * hp.nb_directions # Initialize 16 positive directions at zero 
        negative_rewards =[0] * hp.nb_directions # Initialize 16 negative directions at zero 
    
        # Getting the positive reward in the positive directions => method explore with direction = "positive"     
        for k in range(hp.nb_directions): 
            positive_rewards[k]=explore(env,normalizer,policy, direction = "positive", delta=deltas[k])
            # delta=deltas[k] <=> kième delta in the range 
            
        # Getting the positive reward in the negative directions =>  method explore with direction = "negative"   
        for k in range(hp.nb_directions): 
            negative_rewards[k]=explore(env,normalizer,policy, direction = "negative", delta=deltas[k])
        
        # Gather all the positive/ negative rewards to compute the standard deviation of these rewards
        all_rewards=np.array(positive_rewards+negative_rewards) #concatenate the twol lists in a numpy array
        sigma_r = all_rewards.std()
        
        # Sorting the rollouts by the max (r_pos, r_neg) and selecting the best direction. 
        # 1/ Create a dictionnary with k as  keys : the integer to 0 to 15 
        # with value : maximum reward in positive and negative direction. 
        scores = {k:max(r_pos,r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        # 2/ sorted(scores.keys(), key = lambda x:scores[x]) sort the highest score rewards 
        # [:hp.nb_best_directions] <=> [0 : hp.nb_best_directions]
        order = sorted(scores.keys(), key = lambda x:scores[x])[:hp.nb_best_directions]
        # 3/ Kpositive_reward, negative_reward, deltas of the best direction. 
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Updating our policy => use update method of policy
        policy.update(rollouts,sigma_r) # rollouts of the best directions <=>  rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
      
        # Printing the final reward of the policy after the update  => use explore methode of polic
        # Policy just updated , no direction & no delta (take the default arguments)
        reward_evaluation = explore(env,normalizer,policy) 
        print('Step: ', step, 'Rewards:', reward_evaluation)
        
# Create different folders exp, brs and monitor 
# In monitor : all the video of the AI       
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
  
# Running the main code 
# Create an object hp of the Hp class 
hp=Hp()
# Fix a seed / all the random sampling will be the same => same pertubations
np.random.seed(hp.seed) 
# Connecting the environment to the AI  
env = gym.make(hp.env_name) # self.env_name = '' # environment name   
env = wrappers.Monitor(env, monitor_dir, force = True) # see the env in the video
nb_inputs = env.observation_space.shape[0] # input-size to implement Policy
nb_outputs = env.action_space.shape[0] # ouptput-size to implement Policy 
policy = Policy(nb_inputs, nb_outputs) # create the object policy : the AI
normalizer = Normalizer(nb_inputs) # normalize the state
train(env, policy, normalizer, hp) # train the AI def train(env, policy, normalizer,hp):

# Package to install 
#pip install gym==0.10.5
#pip install pybullet==2.0.8
#conda install -c conda-forge ffmpeg
# the hal-cheetah environment 
# write the environment : HalfCheetahBulletEnv-v0

