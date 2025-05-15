from torch import nn


# --- Dueling Advantage Network ---
class DuelingAdvantageNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(DuelingAdvantageNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Value head: estimates V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Advantage head: estimates A(s, a)
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x):
        features = self.feature(x)
        value = self.value_head(features)             # shape: (batch, 1)
        advantages = self.advantage_head(features)    # shape: (batch, num_actions)
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values, value, advantages