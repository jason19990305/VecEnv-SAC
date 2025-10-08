from torch.distributions import Normal
import torch.nn.functional as F 
import torch.nn as nn 
import torch

epsilon = 1e-6


class Actor(nn.Module):
    def __init__(self, args, hidden_layers=[64, 64]):
        super(Actor, self).__init__()

        self.num_states = args.num_states
        self.num_actions = args.num_actions
        self.action_max = args.action_max

        # Insert input and output sizes into hidden_layers
        hidden_layers.insert(0, self.num_states)

        # Create fully connected layers
        fc_list = []
        for i in range(len(hidden_layers) - 1):
            num_input = hidden_layers[i]
            num_output = hidden_layers[i + 1]            
            layer = nn.Linear(num_input, num_output)
            fc_list.append(layer)
            
        self.mean_linear = nn.Linear(hidden_layers[-1], self.num_actions)
        self.std_linear = nn.Linear(hidden_layers[-1], self.num_actions)
        
        # Convert list to ModuleList for proper registration
        self.layers = nn.ModuleList(fc_list)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Pass input through all layers except the last, applying ReLU activation
        for i in range(len(self.layers)):
            x = self.relu(self.layers[i](x))
        mean = self.mean_linear(x)
        log_std = self.std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean , log_std
    
    def sample(self,state):
        mean , log_std = self.forward(state)
        std = torch.exp(log_std)
        try:
            dist = Normal(mean,std)
        except Exception as e:
            print("mean:", mean)
            print("std:", std)
            for param in self.parameters():
                print("Actor output out of range, check the input state or model parameters.")
                print("actor parameter:", param.data)
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_max
        log_prob = dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_max * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_max
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, args,hidden_layers=[64,64]):
        super(Critic, self).__init__()
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        # add in list
        hidden_layers.insert(0,self.num_states+self.num_actions)
        hidden_layers.append(1)
        print(hidden_layers)

        # create layers
        layer_list = []

        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)            
            layer_list.append(layer)
        # put in ModuleList
        self.layers = nn.ModuleList(layer_list)
        self.relu = nn.ReLU()

    def forward(self,s,a):
        input_data = torch.cat((s,a),dim=1)
        for i in range(len(self.layers)-1):
            input_data = self.relu(self.layers[i](input_data))

        # predicet value
        v_s = self.layers[-1](input_data)
        return v_s