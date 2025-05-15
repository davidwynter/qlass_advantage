# Advantage-Guided QLASS

Advantage-Guided QLASS is an end-to-end system that boosts LLM inference by combining exploration tree methods with a dueling (advantage) network. The system leverages the OpenAI API to drive a configurable language model for action selection. It integrates the following key components:

- **Exploration Tree:**  
  Constructs an exploration tree from an initial state by repeatedly calling the agent’s policy and environment simulator. Each tree node stores a state, the action taken, the immediate reward, and a cumulative reward. Branches are pruned if they do not meet a cumulative reward threshold.

- **Dueling Advantage Network:**  
  Implements a neural network with a shared feature extractor and separate value and advantage heads. This network is trained using a temporal-difference (TD) loss, with the goal of learning a fine-grained advantage function for guiding inference.

- **LLM Policy via OpenAI API:**  
  Replaces a dummy policy network by querying a configurable OpenAI language model (e.g., `o4-mini`) to decide on actions. The current state is converted into a string prompt, and the API’s response is parsed as a discrete action.

- **Boosted Inference:**  
  Once trained, the advantage network is used to guide inference. Actions are selected based on the highest predicted advantage, leading to improved decision-making during multi-step planning.

## Design and Implementation

1. **Exploration Tree with Pruning:**  
   - Uses an environment simulator and calls to the agent’s policy (via the OpenAI API) to generate (state, action, reward, next state) transitions.
   - Prunes branches whose cumulative reward falls below a defined threshold.

2. **Dueling Advantage Network:**  
   - Uses a dueling architecture to compute state value V(s) and action advantage A(s, a) separately.
   - Trains with a TD loss to match predicted Q-values (reconstructed from V(s) and A(s, a)) against targets derived from rewards and discounted future rewards.

3. **LLM Policy Integration:**  
   - Converts a given state (e.g., a NumPy array) into a string prompt.
   - Queries the OpenAI API (e.g., using the `o4-mini` model) to obtain an action decision.
   - Parses and returns the chosen action.

4. **End-to-End Pipeline:**  
   - An initial state is used to build an exploration tree.
   - Transitions are extracted and used to train the dueling advantage network.
   - The trained network then guides inference by selecting actions with the highest advantage.

## Installation

### Prerequisites

- Python 3.8 or later
- An OpenAI API key (set as an environment variable)
- [pip](https://pip.pypa.io/en/stable/) for package installation

### Setup Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/davidwynter/qlass_advantage.git
   cd advantage-qlass
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   If you use a requirements file:
   
   ```bash
   pip install -r requirements.txt
   ```
   
   Or, if you prefer using the PEP 517 build system with the provided `pyproject.toml`:
   
   ```bash
   pip install .
   ```

4. **Configure OpenAI API Key:**

   Set your OpenAI API key as an environment variable:
   
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```
   
   On Windows:
   
   ```bash
   set OPENAI_API_KEY="your_openai_api_key_here"
   ```

## Usage

Run the main application to perform the following steps:
1. Build an exploration tree using the LLM-driven policy and environment simulator.
2. Extract transitions (state, action, reward, next state) from the tree.
3. Train the dueling advantage network on these transitions.
4. Use the trained network to boost inference by guiding action selection.

Execute the main script:

```bash
python main.py
```

### Configuration

- **Exploration Parameters:**  
  Adjust `max_depth` and `prune_threshold` in the code to control the exploration tree depth and branch pruning.

- **Network & Training Hyperparameters:**  
  Modify parameters such as `input_dim`, `hidden_dim`, `num_actions`, `gamma`, and `learning_rate` to suit your problem.

- **LLM API Settings:**  
  In the `agent_policy` function, you can adjust the prompt, model (e.g., `o4-mini`), temperature, and other parameters.

## Project Structure

```
advantage-qlass/
├── main.py              # Main application entry point.
├── README.md            # This file.
├── pyproject.toml       # Project configuration for build and dependency management.
└── requirements.txt     # Optional: list of dependencies.
```

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is inspired by recent work in Q-guided language agent inference and advantage-based reinforcement learning.
```

---

### Additional Notes

- **main.py:**  
  The main application script should integrate all the components (exploration tree, training loop, and boosted inference) as shown in the earlier code examples.

- **requirements.txt (Optional):**  
  You may create a simple `requirements.txt` with the following content if you prefer:
  
  ```
  numpy>=1.18.0
  torch>=1.7.0
  openai>=0.27.0
  ```
