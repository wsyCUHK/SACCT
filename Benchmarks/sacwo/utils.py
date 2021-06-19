import math
import torch
import numpy as np
import itertools
import random

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


##################Here We denote the system functions######################
#########################################################################
bandwidth=2*10**6
N0=10**(-10)
candidate_bitrate=np.array([16,8,5,1])
num_of_bitrate=candidate_bitrate.shape[0]
candidate_channel=np.array([1,2,3])
num_of_channel=candidate_channel.shape[0]
N_S = 2+num_of_bitrate
T=2
kappa1=0.2
kappa2=200
user_qoe=[1.5,1.5,2.5]
#print([loss1_weight,loss2_weight,loss3_weight])
user_para=[[1,0.05,0.5],[2,5,2],[2,5,0.5]]
def user_mod(provider_set,user_type,last_bit,distance):
    #print(user_type)
    eta1,eta2,eta3=user_para[user_type]
    #print(eta1)
    #print(eta2)
    #print("eta1: {}, eta2: {}, eta3: {}".format(eta1, eta2,eta3))
    max_bitrate=0
    max_qoe=-10000
    #provider_set_=
    last=candidate_bitrate[last_bit]
    #print(last)
    #print(provider_set)
    hit=np.random.exponential(1)
    for index in range(provider_set.shape[0]):
        if provider_set[index]>0.5:
            i=candidate_bitrate[index]
            
            cur_qoe=np.power(i,1.012)*eta1-eta2*abs(last_bit-index)-eta3*(i/(distance*hit+0.1))
            #print(i/(15*hit))
            #print(max((i/(15*hit))-2,0))
            #print("Gain: {}, Variation: {}, Delay: {},Total: {}".format(eta1*i, eta2*abs(last_bit-index), eta3*(i/(distance*hit+0.1)),cur_qoe))
            if cur_qoe>max_qoe:
                max_qoe=cur_qoe
                max_bitrate=index
    #print(max_bitrate)
    #print(max_qoe)
    return max_bitrate, max(max_qoe,-50)#max(max_qoe,0)

def qoe_loss(cached_bitrates,request):
    loss=[]
    gain=[]
    for i in range(request.shape[0]):
        small=1000
        gain.append(candidate_bitrate[request[i]])
        for j in range(cached_bitrates.shape[0]):
            if cached_bitrates[j]>0.5:
                if (candidate_bitrate[request[i]]-candidate_bitrate[j])**2< small:
                    small=(candidate_bitrate[request[i]]-candidate_bitrate[j])**2
        loss.append(small)
    return np.sum(loss),1.5*np.sum(gain)


def function_mu(encoding_bitrate,transcoding_bitrate):
    transcoding_cpu_cycle=0
    for j in range(transcoding_bitrate.shape[0]):
        if transcoding_bitrate[j]>0.5:
            transcoding_cpu_cycle+=abs(candidate_bitrate[encoding_bitrate]-candidate_bitrate[j])
    return transcoding_cpu_cycle


def random_actor(num_of_bitrate):
    a=[random.randint(0,num_of_bitrate-1)]
    for i in range(num_of_bitrate):
        a.append(random.randint(0,1))
    return np.array(a)

#def bitrate_request(last_bit):






def cvxpy(s0,s1,transcoding_bitrate):
    para_a=bandwidth*N0*1000.0/(s0*1000.0)
    para_b=candidate_bitrate[int(np.round(s1))]*(10.0**6)*T/bandwidth
    para_k3=function_mu(int(np.round(s1)),transcoding_bitrate)
    value2=[]
    smallest=10.0**20
    index=0
    #The python cvxopt only support the standard form: like quadratic smoothing, total variation
    #As a result, we implement a naive cvx optimizer here
    #The problem is equalvalent to split the time for transcoding and transmission
    #The graident is monotonic
    #So the remianing part is to find the optimal splitting
    if para_k3==0:
        x=T
        smallest=para_a*x*(np.power(2.0,para_b/x-1))/(10.0**5)
        index=x
    else:
        for xg in range(5,95):
            x=T/100*xg
            #print(x)
            xg_value=(para_a*x*(np.power(2.0,para_b/x-1))+kappa2*(np.power(para_k3,3)/(np.power(T-x,2))))/(10.0**5)
            if xg_value<smallest:
                index=x
                smallest=xg_value
            else:
                break
    return smallest

def truc_norm():
    a=np.random.normal(2.6,1)
    if a<0.2:
        a=0.2
    elif a>5:
        a=5
    #print(a)
    return np.int(a)
########################################################################################
#############################Function Defined###########################################
########################################################################################