import torch.nn as nn
import torch.nn.functional as F

class SpriteEncoder(nn.Module):
    def __init__(self,
                 sprite_width, height, channels, output_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_dim*sprite_width*height, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x
