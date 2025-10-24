

import gymnasium as gym # openai gym
import numpy as np 
import argparse
from SAC.Agent import Agent
from gymnasium.wrappers import RecordVideo


class main():
    def __init__(self,args):
        env_name = 'Hopper-v5'
        env = gym.make(env_name)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        args.env_name = env_name
        
        # args
        args.num_actions = num_actions
        args.num_states = num_states
        args.action_max = env.action_space.high[0]  # Pendulum action space is continuous, so we need to normalize it


        # print args 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")

        # create agent
        hidden_layer_num_list = [256,256]
        agent = Agent(args , env , hidden_layer_num_list)

        # trainning
        agent.train() 
        
        # evaluate 
        render_env = gym.make(env_name, render_mode="rgb_array")  
        render_env = RecordVideo(render_env, video_folder = "Video/"+env_name, episode_trigger=lambda x: True)
        agent.evaluate_policy(render_env)
        render_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for SAC")
    parser.add_argument("--d", type=int, default=1, help="Update target network every d step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--tau", type=float, default=0.005, help="Parameter for soft update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--init_alpha", type=float, default=0.2, help="Tempture parameter")
    parser.add_argument("--batch_size", type=int, default=2048 , help="Batch size")
    parser.add_argument("--update_freq_steps", type=int, default=5000, help="Update frequency in steps")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Learning rate of actor")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq_steps", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq_steps' steps")
    args = parser.parse_args()

    main(args)