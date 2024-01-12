import sys
import torch
import torch.nn as nn
import torch_directml
import torch
from torch import nn
from torchvision import datasets, transforms
import torch_directml
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
import matplotlib.pyplot as plt

dml = torch_directml.device()



try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)


def play(env, model):

    seed = 2000
    obs, _ = env.reset(seed=seed)
    
    # drop initial frames
    action0 = 0
    for i in range(50):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    while not done:
        p = model.predict(obs) # adapt to your model
        action = np.argmax(p)  # adapt to your model
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated




env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': 'human'
}

env_name = 'CarRacing-v2'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)


class SimpleCNN(nn.Module):
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # INPUT IMAGE SIZE: 96x96x3
        # CONVOLUTION OUTPUT SIZE FORMULA: (W - K + 2P) / S + 1
        
        # First conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2) # (96 - 5 + 2*2) / 1 + 1 = 96
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (96 - 2) / 2 + 1 = 48

        # Calculate the output size after the first conv and pool layers

        # Second conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # (48 - 3 + 2*1) / 1 + 1 = 48
        # after applying pool: (48 - 2) / 2 + 1 = 24

        # Third conv layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # (24 - 3 + 2*1) / 1 + 1 = 24
        # after applying pool: (24 - 2) / 2 + 1 = 12

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)  # 5 output classes

    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x,1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = SimpleCNN() 
model.load_state_dict(torch.load('simpleCNN_V0.pth'))


# your trained
model = model.to(dml)

play(env, model)


