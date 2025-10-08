
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import torch
import time
import copy

# Custom class
from SAC.ReplayBuffer import ReplayBuffer
from SAC.ActorCritic import Actor , Critic



class Agent():
    def __init__(self,args,env,hidden_layer_num_list=[64,64]):

        # Hyperparameter
        self.evaluate_freq_steps = args.evaluate_freq_steps
        self.max_train_steps = args.max_train_steps
        self.num_actions = args.num_actions
        self.mini_batch_size = args.mini_batch_size
        self.num_states = args.num_states
        self.mem_min = args.mem_min
        self.gamma = args.gamma
        self.init_alpha = args.init_alpha
        self.tau = args.tau
        self.lr = args.lr
        self.d = args.d

        # Variable
        self.total_steps = 0
        self.training_count = 0
        self.evaluate_count = 0

        # other
        self.env = env
        self.action_max = env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(args)
        

        # Actor-Critic
        self.actor = Actor(args,hidden_layer_num_list.copy())
        self.critic1 = Critic(args,hidden_layer_num_list.copy())
        self.critic2 = Critic(args,hidden_layer_num_list.copy())
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.optimizer_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        # Alpha optimizer
        self.log_alpha = torch.tensor(np.log(self.init_alpha))
        self.log_alpha.requires_grad = True
        self.target_entropy = - torch.tensor(self.num_actions, dtype=torch.float)
        self.optimizer_alpha = torch.optim.Adam([self.log_alpha] , lr=self.lr, eps=1e-5)

        print(self.actor)
        print(self.critic1)
        print(self.critic2)
        print("-----------")
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def choose_action(self,state):

        state = torch.tensor(state, dtype=torch.float)

        s = torch.unsqueeze(state,0)
        with torch.no_grad():
            action , _ , _ = self.actor.sample(s)

        return action.cpu().numpy().flatten()

    def evaluate_action(self,state):

        state = torch.tensor(state, dtype=torch.float)

        s = torch.unsqueeze(state,0)
        with torch.no_grad():
            _ , log_prob , action = self.actor.sample(s)

        return action.cpu().numpy().flatten() , log_prob.item()

    def evaluate_policy(self, env , render = False):
        times = 10
        evaluate_reward = 0
        entropy_mean = 0
        step_count = 0
        for i in range(times):
            s, info = env.reset()
            
            done = False
            episode_reward = 0
            while True:
                a , log_prob = self.evaluate_action(s)  # We use the deterministic policy during the evaluating
                entropy_mean += -log_prob
                s_, r, done, truncted, _ = env.step(a)

               
                episode_reward += r
                s = s_
                step_count += 1
                #print(episode_reward)
                if truncted or done:
                    break
            evaluate_reward += episode_reward

        return evaluate_reward / times , entropy_mean / step_count

        
    def train(self):
        time_start = time.time()
        step_reward_list = []
        step_count_list = []
        alpha_list = []
        entropy_list = []
        
        # Training Loop
        while self.total_steps < self.max_train_steps:
            s = self.env.reset()[0]            
            while True:
                a = self.choose_action(s)
                s_, r, done , truncated , _ = self.env.step(a)
                done = done or truncated


                # storage data
                self.replay_buffer.store(s, a, [r], s_, done)
                
                # update state
                s = s_

                if self.replay_buffer.size >= self.mem_min:
                    self.training_count += 1
                    self.update()

                if self.total_steps % self.evaluate_freq_steps == 0:
                    self.evaluate_count += 1
                    evaluate_reward , entropy = self.evaluate_policy(self.env)
                    step_reward_list.append(evaluate_reward)
                    step_count_list.append(self.total_steps)
                    alpha_list.append(self.alpha.item())
                    entropy_list.append(entropy)
                    time_end = time.time()
                    h = int((time_end - time_start) // 3600)
                    m = int(((time_end - time_start) % 3600) // 60)
                    second = int((time_end - time_start) % 60)
                    print("---------")
                    print("Time : %02d:%02d:%02d"%(h,m,second))
                    print("Training \tStep : %d / %d"%(self.total_steps,self.max_train_steps))
                    print("Evaluate count : %d\tEvaluate reward : %0.2f"%(self.evaluate_count,evaluate_reward))

                self.total_steps += 1
                if done or truncated:
                    break

        # Plot the training curve
        plt.plot(step_count_list, step_reward_list)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Training Curve")
        plt.show()
        
        # Plot the alpha curve
        plt.plot(step_count_list, alpha_list)
        plt.xlabel("Steps")
        plt.ylabel("Alpha")
        plt.title("Training Curve")
        plt.show()
        
        # Plot the entropy curve
        plt.plot(step_count_list, entropy_list)
        plt.xlabel("Steps")
        plt.ylabel("Entropy")
        plt.title("Training Curve")
        plt.show()
            
    def update(self):
        minibatch_s, minibatch_a, minibatch_r, minibatch_s_, minibatch_done = self.replay_buffer.sample_minibatch() 

        
        # Get target value (Maximum Entropy)
        with torch.no_grad():
            next_action , next_log_prob , _ = self.actor.sample(minibatch_s_)
            next_value1 = self.critic1_target(minibatch_s_,next_action)
            next_value2 = self.critic2_target(minibatch_s_,next_action)
            next_min_value = torch.min(next_value1 , next_value2)
            target_value = minibatch_r + self.gamma * (next_min_value * (1 - minibatch_done) - self.alpha * next_log_prob)
            
        # Update Critic 1
        value1 = self.critic1(minibatch_s , minibatch_a)
        critic1_loss = F.mse_loss(value1 , target_value)
        self.optimizer_critic1.zero_grad()
        critic1_loss.backward()
        self.optimizer_critic1.step()
        
        # Update Critic 2
        value2 = self.critic2(minibatch_s,minibatch_a)
        critic2_loss = F.mse_loss(value2 , target_value)
        self.optimizer_critic2.zero_grad()
        critic2_loss.backward()
        self.optimizer_critic2.step()
        
        # Update Actor
        action , log_prob , _ = self.actor.sample(minibatch_s)
        value1 = self.critic1(minibatch_s , action)
        value2 = self.critic2(minibatch_s , action)
        min_value = torch.min(value1,value2)
        actor_loss =  (self.alpha.detach() * log_prob - min_value).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # Update alpha
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()        
        self.optimizer_alpha.zero_grad()    
        alpha_loss.backward()
        self.optimizer_alpha.step()
        
        # Update target networks
        if self.total_steps % self.d == 0 :         
            self.soft_update(self.critic1_target,self.critic1, self.tau)
            self.soft_update(self.critic2_target,self.critic2, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    
        