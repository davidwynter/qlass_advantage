import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dummy_sim_env import DummyEnvSimulator

# Import your custom environments.
# For example, if you have a health treatment simulator in health_treatment_env.py:
try:
    from cph_treatment_env import CPTreatmentEnv
except ImportError:
    CPTreatmentEnv = None  # Fallback if custom env is unavailable

# ---------------------------
# Text Embedding Conversion
# ---------------------------
# Load a pre-trained model and tokenizer for converting textual states to embeddings.
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def text_to_embedding(text):
    """
    Convert a textual state into a fixed-size vector embedding.
    Uses mean pooling over token embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # Mean pooling over the sequence dimension.
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0).numpy()


# --- Dueling Advantage Network (from previous design) ---
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
        value = self.value_head(features)                   # shape: (batch, 1)
        advantages = self.advantage_head(features)            # shape: (batch, num_actions)
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values, value, advantages

# --- Agent Policy using OpenAI API ---
import openai
openai.api_key = "your_openai_api_key_here"

def agent_policy(state, num_actions=4):
    """
    Convert a given state (assumed to be convertible to string) to a prompt and call the OpenAI API.
    Returns a discrete action (0,1,..., num_actions-1).
    """
    # Convert the state to string. For a numerical state, we can format it as needed.
    state_str = str(state)
    prompt = (
        f"Given the following state:\n{state_str}\n"
        f"Select the best action from 0, 1, 2, 3. "
        "Return only the integer action."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a decision-making agent for a business healthcare task."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=5,
    )
    action_text = response["choices"][0]["message"]["content"].strip()
    try:
        action = int(action_text)
    except ValueError:
        action = 0
    return action

# --- Exploration Tree Infrastructure ---
class TreeNode:
    def __init__(self, state, action=None, reward=0.0, parent=None):
        self.state = state        # The state (e.g., numerical vector)
        self.action = action      # Action taken to reach this node
        self.reward = reward      # Immediate reward obtained
        self.parent = parent      # Parent node
        self.children = []
        self.cumulative_reward = reward if parent is None else parent.cumulative_reward + reward

    def add_child(self, child_node):
        self.children.append(child_node)

def expand_tree_with_env(node, env, depth, max_depth, prune_threshold):
    """
    Expand the tree using the provided environment instance.
    Here, env is assumed to have a step(action) method that returns (next_state, reward, done, info).
    """
    if depth >= max_depth:
        return
    # Use the agent policy to select an action based on the node's state.
    action = agent_policy(node.state)
    next_state, reward, done, info = env.step(action)
    child_node = TreeNode(next_state, action, reward, parent=node)
    # Prune if cumulative reward is too low or the episode has terminated.
    if child_node.cumulative_reward < prune_threshold or done:
        return
    node.add_child(child_node)
    expand_tree_with_env(child_node, env, depth + 1, max_depth, prune_threshold)

def extract_transitions(node):
    """
    Recursively extract transitions (s, a, r, s') from the exploration tree.
    """
    transitions = []
    for child in node.children:
        transitions.append((node.state, child.action, child.reward, child.state))
        transitions.extend(extract_transitions(child))
    return transitions

# --- Training the Advantage Network ---
def train_advantage_network(transitions, adv_net, optimizer, gamma=0.99, num_epochs=1000, batch_size=16):
    # Convert textual states to embeddings.
    state_embeddings = [text_to_embedding(t[0]) for t in transitions]
    next_state_embeddings = [text_to_embedding(t[3]) for t in transitions]
    states = torch.tensor(np.array(state_embeddings), dtype=torch.float32)
    actions = torch.tensor(np.array([t[1] for t in transitions]), dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(np.array([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_state_embeddings), dtype=torch.float32)
    dataset_size = states.shape[0]
    
    loss_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        indices = np.random.choice(dataset_size, batch_size, replace=True)
        batch_states = states[indices]
        batch_actions = actions[indices]
        batch_rewards = rewards[indices]
        batch_next_states = next_states[indices]
        
        q_values, _, _ = adv_net(batch_states)
        pred_q = q_values.gather(1, batch_actions)
        
        with torch.no_grad():
            next_q_values, _, _ = adv_net(batch_next_states)
            max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
        td_target = batch_rewards + gamma * max_next_q
        
        loss = loss_criterion(pred_q, td_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def boosted_inference(state_text, adv_net):
    adv_net.eval()
    # Convert the textual state to an embedding.
    state_vec = text_to_embedding(state_text)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, _, advantages = adv_net(state_tensor)
        chosen_action = advantages.squeeze(0).argmax().item()
    adv_net.train()
    return chosen_action


# --- Main Application ---
def main():
    parser = argparse.ArgumentParser(description="Advantage-Guided QLASS with Custom Simulator")
    parser.add_argument("--env", type=str, default="health", choices=["health", "dummy"],
                        help="Select the environment simulator to use.")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth for exploration tree.")
    parser.add_argument("--prune_threshold", type=float, default=0.3, help="Pruning threshold for cumulative reward.")
    args = parser.parse_args()

    # Select environment based on argument.
    if args.env == "health":
        if CPTreatmentEnv is None:
            raise ImportError("CPTreatmentEnv module not found.")
        env = CPTreatmentEnv()  # Custom simulator for CP Q&A tasks.
        # For training the network, assume the state can be converted to a numerical vector.
        # You might need to add an embedding conversion if the state is textual.
    else:
        # Fallback to a dummy environment simulator.
        from dummy_env import DummyEnvSimulator  # Assume you have this module.
        env = DummyEnvSimulator()

    # Reset the environment to get the initial state.
    init_state = env.reset()
    print("Initial State:", init_state)
    
    # Create root node for exploration tree.
    root = TreeNode(init_state)
    expand_tree_with_env(root, env, depth=0, max_depth=args.max_depth, prune_threshold=args.prune_threshold)
    
    # Extract transitions from the exploration tree.
    transitions = extract_transitions(root)
    print(f"Extracted {len(transitions)} transitions from the exploration tree.")
    
    # Initialize the Advantage Network.
    # Here, input_dim should match the state vector dimension.
    # If using a textual state (e.g., from health simulator), you may need to convert it to a fixed-size vector.
    input_dim = 10  # Adjust based on your actual state representation.
    hidden_dim = 32
    num_actions = 4
    adv_net = DuelingAdvantageNet(input_dim, hidden_dim, num_actions)
    optimizer = optim.Adam(adv_net.parameters(), lr=1e-3)
    
    if len(transitions) > 0:
        train_advantage_network(transitions, adv_net, optimizer, num_epochs=1000, batch_size=16)
    else:
        print("No transitions extracted; adjust exploration parameters.")
    
    # Boosted inference: use the trained network to select an action given a new state.
    test_state = env.reset()
    print("Test State:", test_state)
    action_boosted = boosted_inference(test_state, adv_net)
    print("Boosted inference selected action:", action_boosted)

if __name__ == "__main__":
    main()
