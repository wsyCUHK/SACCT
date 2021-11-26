import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import pickle
class GELU(torch.nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        try:
            torch.nn.init.constant_(m.bias, 0)
        except:
            print("No Bias")

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.active=GELU()
        self.apply(weights_init_)

    def forward(self, state):
        x = self.active(self.linear1(state))
        x =self.active(self.linear2(x))
        x = self.linear3(x)
        return x



class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, nhead=1,nlayer=1):
        super(QNetwork, self).__init__()

        self.head_dim=int(hidden_dim/4)


        

        # Q1 architecture
        self.embeds_state = nn.Embedding(num_inputs, hidden_dim)
        #self.embeds_action= nn.Embedding(num_actions,hidden_dim)
        self.embeds_action= nn.Linear(num_actions,hidden_dim,bias=False)
        #self.embeds_channel=nn.Linear(int(num_inputs/6),hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=6, nhead=nhead)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        #print('Layers: '+str(head_size))
        #self.linear1d = nn.Linear(num_actions, self.head_dim)
        self.linear2 = nn.Linear(num_inputs+hidden_dim, hidden_dim)
        #print(6*self.head_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.embeds_state_ = nn.Embedding(num_inputs, hidden_dim)
        #self.embeds_action_= nn.Embedding(num_actions,hidden_dim)
        self.embeds_action_= nn.Linear(num_actions,hidden_dim,bias=False)
        #self.embeds_channel_=nn.Linear(int(num_inputs/6),hidden_dim)

        #self.encoder_layer4 = nn.TransformerEncoderLayer(d_model=6, nhead=6)
        self.transformer_encoder4 = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.linear5 = nn.Linear(num_inputs+hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)


        self.apply(weights_init_)
        self.active=GELU()
        #print(hidden_dim)
        
        
        self.ffn_norm1 = nn.LayerNorm(int(num_inputs/6), eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(int(num_inputs/6), eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(int(num_inputs/6), eps=1e-6)
        self.ffn_norm4 = nn.LayerNorm(int(num_inputs/6), eps=1e-6)
        self.ffn_norm5 = nn.LayerNorm(int(num_inputs/6), eps=1e-6)
        self.ffn_norm6 = nn.LayerNorm(int(num_inputs/6), eps=1e-6)

        self.pos_encoder = PositionalEncoding(6)
        self.ffn_norm7 = nn.LayerNorm(num_inputs, eps=1e-6)
        self.ffn_norm8 = nn.LayerNorm(num_inputs, eps=1e-6)
        device = torch.device("cuda" if True else "cpu")
        self.action_one_hot=torch.zeros([num_actions,1]).to(device)
        for i in range(num_actions):
            self.action_one_hot[num_actions-1-i,0]=2**i
        self.pool=torch.nn.AvgPool1d(3, stride=2,padding=1)

    def forward(self, state, action):
        #xu = torch.cat([state, action], 1)
        num_inputs=state.shape[1]
        new_state=[]
        new_state.append(self.ffn_norm1(state[:,int(num_inputs/6)*0:int(num_inputs/6)*(1)]))
        new_state.append(self.ffn_norm2(state[:,int(num_inputs/6)*1:int(num_inputs/6)*(2)]))
        new_state.append(self.ffn_norm3(state[:,int(num_inputs/6)*2:int(num_inputs/6)*(3)]))
        new_state.append(self.ffn_norm4(state[:,int(num_inputs/6)*3:int(num_inputs/6)*(4)]))
        new_state.append(self.ffn_norm5(state[:,int(num_inputs/6)*4:int(num_inputs/6)*(5)]))
        new_state.append(self.ffn_norm6(state[:,int(num_inputs/6)*5:int(num_inputs/6)*(6)]))
        xu = torch.cat([new_state[0].unsqueeze(-1),new_state[1].unsqueeze(-1),new_state[2].unsqueeze(-1),new_state[3].unsqueeze(-1),new_state[4].unsqueeze(-1),new_state[5].unsqueeze(-1)], 2)

        #with open('filename.pickle', 'wb') as handle:
        #    pickle.dump([state,xu], handle, protocol=pickle.HIGHEST_PROTOCOL)
        #torch.save([state,xu], 'tensor.pt')
        xu_flatten=torch.cat([new_state[0],new_state[1],new_state[2],new_state[3],new_state[4],new_state[5]], 1)
        #print(bitrate_history.shape)
        #embed_s=self.embeds_state(torch.matmul(bitrate_history,self.action_one_hot).long()).squeeze()
        #context1=self.attention1(new_state[0])[1].squeeze()
        #print(embed_s.shape)
        #print(torch.mm(action,self.action_one_hot).shape)

        embed_a=self.embeds_action(action)
        #print(embed_a.shape)
        #embed_c=self.embeds_channel(channel)
        #print(xu.shape)
        embed_s=self.transformer_encoder1(self.pos_encoder(xu.transpose(0,1)))
        #print(embed_s.shape)
        embed_s=embed_s.transpose(0,1)
        #print(embed_s.shape)
        #embed_s=self.pool(embed_s)
        #print(embed_s.shape)
        embed_s=torch.flatten(embed_s, start_dim=1)
        #print(embed_s.shape)
        #print(embed_a.shape)
        #print(embed_c.shape)
        #embed_s=self.ffn_norm7(embed_s+xu_flatten)
        x1=torch.cat([embed_s,embed_a], 1)
        #print(x1d.shape)
        x1 = self.active(self.linear2(x1))
        x1 =self.linear3(x1)

        
        

        embed_a_=self.embeds_action_(action)
        embed_s_=self.transformer_encoder4(self.pos_encoder(xu.transpose(0,1)))
        #print(embed_s.shape)
        embed_s_=embed_s_.transpose(0,1)
        #print(embed_s.shape)
        #embed_s_=self.pool(embed_s_)
        embed_s_=torch.flatten(embed_s_, start_dim=1)
        #print(embed_s.shape)
        #print(embed_a.shape)
        #print(embed_c.shape)
        #embed_s_=self.ffn_norm8(embed_s_+xu_flatten)
        x2=torch.cat([embed_s_,embed_a_], 1)
        x2 = self.active(self.linear5(x2))
        x2 = self.active(self.linear6(x2))

        #print(x1)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            #tanh function output is [-1,1], as a result we need to divided by 2.0
            # self.action_scale = torch.FloatTensor(
            #     (action_space.high - action_space.low) / 2.)
            # self.action_bias = torch.FloatTensor(
            #     (action_space.high + action_space.low) / 2.)
            self.action_scale=torch.FloatTensor((action_space-1)/2.)
            self.action_bias=torch.FloatTensor((action_space-1)/2.)

    #Generate the guassian actor paremeter mean and sigma
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) 
        # Return the probablity of each action under current normal distribution
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)



class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        #the initialization is exactly the same as the gaussian policy
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x =  self.active(self.linear1(state))
        x =  self.active(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean


    def sample(self, state):
        #Directly add noise/randomness at the action
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
