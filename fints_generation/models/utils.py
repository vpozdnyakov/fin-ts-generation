from torch.utils.data import Dataset
import torch

class SlidingWindowDataset(Dataset):
    def __init__(self, df, window_size, step_size):
        self.window_size = window_size
        self.df = df
        self.step_size = step_size
    
    def __len__(self):
        return (len(self.df) - self.window_size + 1) // self.step_size
    
    def __getitem__(self, idx):
        target = self.df.iloc[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        return torch.FloatTensor(target.values)
