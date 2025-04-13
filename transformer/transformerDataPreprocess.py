class IMUDataset(Dataset):
    def __init__(self, features, labels, seq_length=200, transform=None):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        self.transform = transform
        
    def __len__(self):
        return len(self.features) - self.seq_length + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]
        y = self.labels[idx+self.seq_length-1]  # Label is for the last time step
        
        if self.transform:
            x = self.transform(x)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Data augmentation transforms
class RandomNoise:
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level
        
    def __call__(self, sample):
        noise = np.random.normal(0, self.noise_level, sample.shape)
        return sample + noise

class RandomTimeWarp:
    def __init__(self, max_time_warp=0.2):
        self.max_time_warp = max_time_warp
        
    def __call__(self, sample):
        t = np.arange(len(sample))
        warp_factor = 1.0 + np.random.uniform(-self.max_time_warp, self.max_time_warp)
        warped_t = np.power(t / len(sample), warp_factor) * len(sample)
        warped_sample = np.zeros_like(sample)
        
        for i in range(sample.shape[1]):
            warped_sample[:, i] = np.interp(t, warped_t, sample[:, i])
            
        return warped_sample
