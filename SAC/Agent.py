
from gymnasium.vector import AsyncVectorEnv
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np 
import torch
import time
import copy
import os


# Custom class
from SAC.ReplayBuffer import ReplayBuffer
from SAC.ActorCritic import Actor , Critic



class Agent():
    def __init__(self,args,env,hidden_layer_num_list=[64,64]):

        # Hyperparameter
        self.evaluate_freq_steps = args.evaluate_freq_steps
        self.update_freq_steps = args.update_freq_steps
        self.max_train_steps = args.max_train_steps
        self.num_actions = args.num_actions
        self.batch_size = args.batch_size
        self.num_states = args.num_states
        self.init_alpha = args.init_alpha
        self.env_name = args.env_name
        self.gamma = args.gamma
        self.tau = args.tau
        self.lr = args.lr
        self.d = args.d

        # Variable
        self.total_steps = 0
        self.training_count = 0
        self.evaluate_count = 0

        # other
        self.env = env
        self.env_eval = copy.deepcopy(env)
        self.num_envs = os.cpu_count() - 1
        print("num_envs : ", self.num_envs)
        self.action_max = env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and self.device.type == "cuda":
            print("Device name : ",torch.cuda.get_device_name(self.device))
        env_fns = [lambda : gym.make(self.env_name) for _ in range(self.num_envs)]
        self.venv = AsyncVectorEnv(env_fns , autoreset_mode= gym.vector.AutoresetMode.SAME_STEP)

        

        # Actor-Critic        
        self.actor = Actor(args,hidden_layer_num_list.copy()).to(self.device)
        self.critic1 = Critic(args,hidden_layer_num_list.copy()).to(self.device)
        self.critic2 = Critic(args,hidden_layer_num_list.copy()).to(self.device)
        self.actor_cpu = Actor(args,hidden_layer_num_list.copy())
        self.critic1_cpu = Critic(args,hidden_layer_num_list.copy())
        self.critic2_cpu = Critic(args,hidden_layer_num_list.copy())
        self.update_cpu()
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
    
    def update_cpu(self):
        self.actor_cpu.load_state_dict(self.actor.state_dict())
        self.critic1_cpu.load_state_dict(self.critic1.state_dict())
        self.critic2_cpu.load_state_dict(self.critic2.state_dict())
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def choose_action(self,state):

        state = torch.tensor(state, dtype=torch.float)
        #s = torch.unsqueeze(state,0)
        with torch.no_grad():
            action , _ , _ = self.actor_cpu.sample(state)

        return action.cpu().numpy()

    def evaluate_action(self,state):

        state = torch.tensor(state, dtype=torch.float)

        s = torch.unsqueeze(state,0)
        with torch.no_grad():
            _ , log_prob , action = self.actor_cpu.sample(s)

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
        evaluate_count = 0
        
         # Reset Vector Env
        s , infos = self.venv.reset()                        

        # Training Loop
        while self.total_steps < self.max_train_steps:            
            # Sample data
            for step in range(self.update_freq_steps // self.num_envs + 1):
                # Choose action
                a = self.choose_action(s)
                s_ , r , done, truncated, infos = self.venv.step(a)  # Vector Env
                # Handle final state
                for j in range(self.num_envs):
                    if done[j] or truncated[j]:
                        next_state = infos["final_obs"][j]
                    else : 
                        next_state = copy.deepcopy(s_[j])                        

                    # s, a, r, s_, done
                    self.replay_buffer.store(s[j], a[j], [r[j]], next_state, [truncated[j] or done[j]])
                    self.total_steps += 1
                    evaluate_count += 1
                
                s = s_
            
            
            
            # Evaluate
            if evaluate_count >= self.evaluate_freq_steps:
                evaluate_reward , entropy = self.evaluate_policy(self.env_eval)
                step_reward_list.append(evaluate_reward)
                step_count_list.append(self.total_steps)
                entropy_list.append(entropy)
                alpha_list.append(self.alpha.item())
                time_end = time.time()
                h = int((time_end - time_start) // 3600)
                m = int(((time_end - time_start) % 3600) // 60)
                second = int((time_end - time_start) % 60)
                print("---------")
                print("Time : %02d:%02d:%02d"%(h,m,second))
                print("Step : %d / %d\tEvaluate reward : %0.2f"%(self.total_steps,self.max_train_steps,evaluate_reward))
                evaluate_count = 0
                
            # Sync
            self.replay_buffer.sync_to_device()
            # Update 
            for _ in range(self.update_freq_steps):
                self.update()
            self.update_cpu()

        plot_dir = "Plot"
        os.makedirs(plot_dir, exist_ok=True)

        # Plot the training curve
        plt.plot(step_count_list, step_reward_list)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Training Curve")
        plt.savefig(os.path.join(plot_dir, f"{self.env_name}_training_curve.png"))
        plt.close()
        
        # Plot the alpha curve
        plt.plot(step_count_list, alpha_list)
        plt.xlabel("Steps")
        plt.ylabel("Alpha")
        plt.title("Alpha Curve")
        plt.savefig(os.path.join(plot_dir, f"{self.env_name}_alpha_curve.png"))
        plt.close()
        
        # Plot the entropy curve
        plt.plot(step_count_list, entropy_list)
        plt.xlabel("Steps")
        plt.ylabel("Entropy")
        plt.title("Entropy Curve")
        plt.savefig(os.path.join(plot_dir, f"{self.env_name}_entropy_curve.png"))
        plt.close()
    
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
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
        self.optimizer_critic1.step()
        
        # Update Critic 2
        value2 = self.critic2(minibatch_s,minibatch_a)
        critic2_loss = F.mse_loss(value2 , target_value)
        self.optimizer_critic2.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
        self.optimizer_critic2.step()
        
        # Update Actor
        action , log_prob , _ = self.actor.sample(minibatch_s)
        value1 = self.critic1(minibatch_s , action)
        value2 = self.critic2(minibatch_s , action)
        min_value = torch.min(value1,value2)
        actor_loss =  (self.alpha.detach() * log_prob - min_value).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer_actor.step()
        
        # Update alpha
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()        
        self.optimizer_alpha.zero_grad()    
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 0.5)
        self.optimizer_alpha.step()
        
        # Update target networks
        if self.total_steps % self.d == 0 :         
            self.soft_update(self.critic1_target,self.critic1, self.tau)
            self.soft_update(self.critic2_target,self.critic2, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            