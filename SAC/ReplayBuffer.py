import numpy as np 
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(args.batch_size)
        self.max_length = int(args.buffer_size)
        self.size = 0
        self.ptr = 0
        self.ptr_dev = 0
        self.num_states = int(args.num_states)
        self.num_actions = int(args.num_actions)

        # store on CPU as numpy arrays to avoid per-step device copies
        self.s = np.zeros((self.max_length, self.num_states), dtype=np.float32)
        self.a = np.zeros((self.max_length, self.num_actions), dtype=np.float32)
        self.r = np.zeros((self.max_length, 1), dtype=np.float32)
        self.s_ = np.zeros((self.max_length, self.num_states), dtype=np.float32)
        self.done = np.zeros((self.max_length, 1), dtype=np.float32)

        # device-side tensors (populated by sync_to_device)
        self.s_dev = None
        self.a_dev = None
        self.r_dev = None
        self.s__dev = None
        self.done_dev = None

    def store(self, s, a, r, s_, done):
        # accept numpy arrays / lists / scalars; assign on CPU
        self.s[self.ptr] = np.asarray(s, dtype=np.float32)
        self.a[self.ptr] = np.asarray(a, dtype=np.float32)
        self.s_[self.ptr] = np.asarray(s_, dtype=np.float32)
        self.r[self.ptr] = np.asarray(r, dtype=np.float32).reshape(-1)[:1]
        self.done[self.ptr] = np.asarray(done, dtype=np.float32).reshape(-1)[:1]

        self.ptr = (self.ptr + 1) % self.max_length
        self.size = min(self.size + 1, self.max_length)

        # mark device buffers stale (will be re-synced before update)
        self._device_synced = False

    def sync_to_device(self):
        
        if self.size == 0:
            return

        # allocate device tensors if not yet
        if self.s_dev is None:
            self.s_dev = torch.empty((self.max_length, self.num_states), dtype=torch.float32, device=self.device)
            self.a_dev = torch.empty((self.max_length, self.num_actions), dtype=torch.float32, device=self.device)
            self.r_dev = torch.empty((self.max_length, 1), dtype=torch.float32, device=self.device)
            self.s__dev = torch.empty((self.max_length, self.num_states), dtype=torch.float32, device=self.device)
            self.done_dev = torch.empty((self.max_length, 1), dtype=torch.float32, device=self.device)

        
        # copy only the active prefix (0:self.size)
        self.s_dev[:self.size].copy_(torch.as_tensor(self.s[:self.size], device=self.device))
        self.a_dev[:self.size].copy_(torch.as_tensor(self.a[:self.size], device=self.device))
        self.r_dev[:self.size].copy_(torch.as_tensor(self.r[:self.size], device=self.device))
        self.s__dev[:self.size].copy_(torch.as_tensor(self.s_[:self.size], device=self.device))
        self.done_dev[:self.size].copy_(torch.as_tensor(self.done[:self.size], device=self.device))

        self.ptr_dev = self.size
        self._device_synced = True

    def sample_minibatch(self):
        # sample indices on CPU
        index = torch.randint(0, self.size, (self.batch_size,))

        if self._device_synced:
            s = self.s_dev[index]
            a = self.a_dev[index]
            r = self.r_dev[index]
            s_ = self.s__dev[index]
            done = self.done_dev[index]
            return s, a, r, s_, done    
        else:
            s = torch.as_tensor(self.s[index.numpy()], device=self.device)
            a = torch.as_tensor(self.a[index.numpy()], device=self.device)
            r = torch.as_tensor(self.r[index.numpy()], device=self.device)
            s_ = torch.as_tensor(self.s_[index.numpy()], device=self.device)
            done = torch.as_tensor(self.done[index.numpy()], device=self.device)
            return s, a, r, s_, done
        