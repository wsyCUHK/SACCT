import argparse
import datetime
#import gym
import numpy as np
import itertools
import torch
import random
import math
#from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from utils import user_mod,qoe_loss,function_mu,random_actor,cvxpy,truc_norm

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: False)')
parser.add_argument('--history_length', type=int, default=3, metavar='N',
                    help='the state length')
parser.add_argument('--filename', type=str, default='test', help='save_file_name')
parser.add_argument('--num_header', type=int, default='1')
parser.add_argument('--num_layer', type=int, default='1')
parser.add_argument('--arrival_rate', type=float, default='1.0')
parser.add_argument('--penalty', type=float, default='1.0')
parser.add_argument('--GPU', type=int, default='2')
args = parser.parse_args()

import os
random.seed(args.seed)
np.random.seed(args.seed)
#os.environ["CUDA_VISIBLE_DEVICES"]=str(random.randint(0,5))
F_c = 915*1e6  # carrier bandwidth
A_d = 79.43  # antenna gain
degree = 2  # path loss value
light = 3*1e8  # speed of light
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP_STEP = 201
computation_upper_bound=2
bandwith_upper_bound=2
P=1000
N0=10**(-10)
candidate_bitrate=np.array([16,8,5,1])
num_of_bitrate=candidate_bitrate.shape[0]
candidate_channel=np.array([1,2,3])
num_of_channel=candidate_channel.shape[0]
N_S = 2+num_of_bitrate
history_length=args.history_length
N_S=history_length*N_S
N_A=num_of_bitrate+1
Computation_Cost=np.array([0,1,2,6,0])
a=0.01
b=0.05
c=[0]
loss2_penalty=0.5
beta=2
N=30
distv = 10+np.random.random_sample((N,1)) * 60
bandwidth=2*10**6
phone_arrival_rate=0.1*args.arrival_rate
phone_departure_rate=0.1#*args.arrival_rate
phone_self_transition=0.8
TV_arrival_rate=0.05
TV_departure_rate=0.05#*args.arrival_rate
TV_self_transition=0.8
laptop_arrival_rate=0.1
laptop_departure_rate=0.1#*args.arrival_rate
laptop_self_transition=0.8
a_rate=[phone_arrival_rate,TV_arrival_rate,laptop_arrival_rate]
#print(a_rate)
d_rate=[phone_departure_rate,TV_departure_rate,laptop_departure_rate]
s_tran=[phone_self_transition,TV_self_transition,laptop_self_transition]
T=2
kappa1=0.2
kappa2=200
loss1_weight=1
#loss2_weight=(3/18) * args.penalty
#loss3_weight=(14/57.12) * args.penalty
user_qoe=[1.5,1.5,2.5]
#print([loss1_weight,loss2_weight,loss3_weight])
user_para=[[1,0.05,0.5],[2,5,2],[2,5,0.5]]


########################################################################################
#############################The environment for the video streaming####################
########################################################################################
def user_transit(users,provider_set):
    new_users=[]
    cur_qoe=0
    for i in range(users.shape[0]):
        if random.random()>d_rate[users[i,1]]:
            #the user doesnot departure
            #new_u=
            bit,qoe=user_mod(provider_set,users[i,1],users[i,0],users[i,2])
            #print(qoe)
            new_users.append([bit,users[i,1],users[i,2]])
            cur_qoe+=qoe
            #New Arrival
    for i in range(3):
        if random.random()<a_rate[i]:
            distance=truc_norm()
            bit,qoe=user_mod(provider_set,i,random.randint(0,num_of_bitrate-1),distance)
            new_users.append([bit,i,distance])
            cur_qoe+=qoe
            #new_users.append([,i])
    return np.array(new_users),cur_qoe

def env(a,s,users):
    ####################Map action in caterogrial##########################################
    caterogrial_action=np.round(a)
    caterogrial_action=caterogrial_action.astype('int')
    state=np.zeros((2+num_of_bitrate,))
    state[0]=s[0]
    #print(state[0])
    state[1]=s[args.history_length]
    for i in range(num_of_bitrate):
        state[2+i]=s[(2+i)*args.history_length]
    encoding_bitrate=caterogrial_action[0]
    transcoding_bitrate=caterogrial_action[1:]
    for i in range(encoding_bitrate):
        transcoding_bitrate[i]=0
    try:
        transcoding_bitrate[encoding_bitrate]=1
    except:
        encoding_bitrate=0
        transcoding_bitrate=np.zeros(num_of_bitrate,)
        transcoding_bitrate[0]=1

    ####################Model User Arrival, Departure, QoE Gain, and  Encoding Energy Cost#################
    #print("Encoding:{},Transcode:{}".format(encoding_bitrate,transcoding_bitrate))
    #print()

    new_users,loss1=user_transit(users,transcoding_bitrate)
    #print(loss1)
    new_users=np.array(new_users)
    #print(new_users)
    loss2=min(candidate_bitrate[encoding_bitrate]*kappa1*T,100)
    ##############Convex Optimization#############################################
    loss3=min(cvxpy(state[0],state[1],transcoding_bitrate),100)
    
    ##############Wireless Channel Model##########################################
    channel_d= np.random.exponential() * A_d * (light / (4.0 * 3.141592653589793 * F_c * distv[0]))**degree
    state_2=np.zeros((num_of_bitrate*args.history_length,))
    
    for i in range(num_of_bitrate):
        state_2[(1+num_of_bitrate*args.history_length):(num_of_bitrate+1)*args.history_length]=s[(2+num_of_bitrate)*args.history_length:((3+num_of_bitrate)*args.history_length-1)]
    for i in range(new_users.shape[0]):
        state_2[new_users[i,0]*args.history_length]=+1

    state_0=np.zeros((args.history_length,))
    state_0[0]=channel_d
    state_0[1:]=s[:args.history_length-1]
    state_1=np.zeros((args.history_length,))
    state_1[0]=encoding_bitrate
    state_1[1:]=s[args.history_length:2*args.history_length-1]
    #print(-(loss1_weight*loss1+loss2_weight*loss2+loss3_weight*loss3))
    #print(loss1)
    #print(loss2)
    #print(loss3)
    #print("Loss1: {}, loss2: {}, loss3: {}".format(loss1_weight*loss1, loss2_weight*loss2, loss3_weight*loss3))
    #print("Loss1: {}, loss2: {}, loss3: {}".format(loss1, loss2, loss3))
    #print("Reward: {},Loss2: {},Loss3: {}, State1:{}, Transcode:{}".format(loss1,loss2,loss3,state[1],transcoding_bitrate))
    return (loss1-loss2-loss3),np.concatenate((np.concatenate((state_0,state_1),axis=0),state_2),axis=0),new_users,loss1,loss2,loss3

def env_reset():
    # Generate the channel gain for next time slot
    channel_d= np.random.exponential() * A_d * (light / (4 * 3.141592653589793 * F_c * distv[0]))**degree
    #s=np.array([channel_d,random.randint(0,3),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)])
    s=np.zeros((N_S,))
    s[0]=channel_d
    s[1]=random.randint(0,3)
    if random.random()<0.01:
        u=np.expand_dims(np.array([random.randint(0,3),random.randint(0,2),truc_norm()]), axis=0)
    elif random.random()<0.1:
        u=np.array([[random.randint(0,3),random.randint(0,2),truc_norm()],[random.randint(0,3),random.randint(0,2),truc_norm()]])
    else:
        u=np.array([[random.randint(0,3),random.randint(0,2),truc_norm()],[random.randint(0,3),random.randint(0,2),truc_norm()],[random.randint(0,3),random.randint(0,2),truc_norm()]])
    for i in range(history_length):
        action=np.array([random.randint(0,3),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)])
        _, s_, u_ ,_,_,_= env(action,s.squeeze(),u)
        s=s_
        u=u_
    return action,s,u

########################################################################################
#############################Environment Done###########################################
########################################################################################

# Agent
#Initialization?
#action=np.array([1,1,1,0,1])
action_space=np.zeros((N_A,))
action_space[0]=num_of_bitrate
for i in range(num_of_bitrate):
    action_space[1+i]=2

#print(args.cuda)
#Tesnorboard
#Log output
#writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'Video Streaming GLN',
##                                                            args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
def herustic_actor(s):
    action=np.ones((1+num_of_bitrate,))
    action[0]=0
    #action[2]=1
    #action[-2]=1
    return action


plot_log=[]
#plot_real=[]
plot_test=[]
step_log=[]
energy_con_log=[]
gain_log=[]
bitrate_request_log=[]
for i_episode in itertools.count(1):
    ###Training Initialization
    episode_reward = 0
    #real_reward=0
    episode_steps = 0
    done = False
    #state = env.reset()
    ################################
    #       Initial State          #
    ################################
    # action=np.array([1,1,1,0,1])
    # state=np.array([1.5759179851703167e-06,1,1,1,1,1])#.reshape((1,2+num_of_bitrate))#.unsqueeze(0)
    # #print(s.shape)
    # #print(s.squeeze().shape)
    # u=np.expand_dims(np.array([1,0]), axis=0)
    action, state,u=env_reset()
    ###############################

    while not done:
        #if args.start_steps > total_numsteps:
            #action = env.action_space.sample()  # Sample random action
            #if random.random()<1.0/np.sqrt(np.sqrt(total_numsteps+1)):
                #action =random_actor(num_of_bitrate)
            #else:
            #    action = agent.select_action(state)
        #else:
            #action = agent.select_action(state)  # Sample action from policy
        action=herustic_actor(state)

        reward, next_state, u_ ,l1,l2,l3= env(action,state.squeeze(),u) # Step
        #print(reward)
        #real,_,_,_,_,_=env(action,state.squeeze(),u)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        #real_reward+=real
        energy_con_log.append(l2+l3)
        gain_log.append(l1)
        step_log.append(reward)
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        #print(episode_steps)
        if episode_steps == MAX_EP_STEP:
            mask = 1
            done =True
        else:
            mask=float(not done)
        #print(state)
        #print(action)
        #print(reward)
        #print(next_state)
        #print(mask)
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        u=u_
        #try:
        #    bitrate_request_log.append([u[:,0]])
        #except:
            #print('empty')


    if total_numsteps > args.num_steps:
        break

    #writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, np.round(episode_reward, 2)))
    plot_log.append(np.round(episode_reward, 2))
    #plot_real.append(round(real_reward, 2))



import pickle
#with 
#with open('gg.pickle', 'wb') as f:
#    pickle.dump(bitrate_request_log, f)


import os


file_id=0
file_exists=True
while file_exists:
    if os.path.isfile(args.filename+str(file_id)+'.pickle'):
        print('File Exists')
        file_id+=1
    else:
        with open(args.filename+str(file_id)+'.pickle', 'wb') as f:
            pickle.dump([step_log,gain_log,energy_con_log], f)
        file_exists=False

avg_reward = 0.
avg_reward2 = 0.

import time
random.seed(time.time())
np.random.seed(int(time.time()))


with torch.no_grad():

    action, state,u=env_reset()
    episode_reward = 0
    #episode_reward2=0
    episode_steps = 0
    done = False
    show_channel=[]
    show_user_number=[]
    show_encoding_rate=[]
    show_transcoding=[]
    qoe_mismatch=[]
    encoding_power=[]
    ttpower=[]
    request1=np.zeros((1000,num_of_bitrate))
    while not done:
        #print(state)
        action = herustic_actor(state)
        #print(action)
        reward, next_state, u_,l1,l2,l3 = env(action,state.squeeze(),u)
        #conti, _, _,_,_,_ = env(action,state.squeeze(),u)
        episode_reward += reward
        #episode_reward2+=conti
        show_channel.append(state[0])
        show_user_number.append(u_.shape[0])
        show_encoding_rate.append(3-action[0])
        qoe_mismatch.append(l1)
        encoding_power.append(l2)
        ttpower.append(l3)
        temp=action[1:]
        for transcoding_transformer in range(int(action[1]+1)):
            temp[transcoding_transformer]=0
        #for transcoding_transformer in range(int(state[1]),num_of_bitrate):
            #temp.append(1)
        #temp[int(state[1])]=0
        show_transcoding.append(temp)

        #print(u_)
        #request_temp=np.zeros((1,num_of_bitrate))
        for i in range(u_.shape[0]):
            request1[episode_steps,u_[i,0]]+=1
        #print(request1[episode_steps,:])
        #request1.append(request_temp)
        state = next_state
        u=u_
        episode_steps+=1
        if episode_steps == 1000:
            done =True
    
    avg_reward += episode_reward

file_id=0
file_exists=True
while file_exists:
    if os.path.isfile(args.filename+str(file_id)+'_detail.pickle'):
        print('File Exists')
        file_id+=1
    else:
        with open(args.filename+str(file_id)+'_detail.pickle', 'wb') as f:
            pickle.dump([show_channel,show_user_number,show_encoding_rate,show_transcoding,qoe_mismatch,encoding_power,ttpower,request1], f)
        file_exists=False