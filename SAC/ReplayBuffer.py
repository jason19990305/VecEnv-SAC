import numpy as np 
import torch




class ReplayBuffer:
    def __init__(self, args):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mini_batch_size = args.mini_batch_size
        self.max_length = args.buffer_size
        self.size = 0       
        self.ptr = 0 
        self.s = torch.zeros((self.max_length, args.num_states))
        self.a = torch.zeros((self.max_length, args.num_actions))
        self.r = torch.zeros((self.max_length, 1))
        self.s_ = torch.zeros((self.max_length, args.num_states))
        self.done = torch.zeros((self.max_length, 1))

    def store(self, s, a, r, s_, done):
        
        self.s[self.ptr] = torch.from_numpy(s)
        self.a[self.ptr] = torch.from_numpy(a)
        self.s_[self.ptr] = torch.from_numpy(s_)
        self.r[self.ptr] = torch.from_numpy(np.array(r))
        self.done[self.ptr] = torch.from_numpy(np.array(done))

        self.ptr = (self.ptr + 1) % self.max_length
        self.size = min(self.size + 1, self.max_length)
  
    def sample_minibatch(self):
        index = torch.randint(0, self.size, (self.mini_batch_size,))
        s = self.s[index]
        a = self.a[index]
        r = self.r[index]
        s_ = self.s_[index]
        done = self.done[index]
        
        return s, a, r, s_, done

