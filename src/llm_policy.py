import openai
import numpy as np

# Set your OpenAI API key (ensure you have installed the openai package)
openai.api_key = "your_openai_api_key_here"

# Number of discrete actions available (must match your environment)
num_actions = 4

def agent_policy(state, num_actions=num_actions):
    """
    Given a state (which is converted to a string representation), query the OpenAI API
    to select an action from {0, 1, ..., num_actions-1}. The API is prompted to output
    a single integer corresponding to the chosen action.
    """
    # Convert the state (e.g., a NumPy array) to a string.
    state_str = np.array2string(state, precision=2, separator=',')
    
    # Construct the prompt.
    prompt = (
        f"Given the following state:\n{state_str}\n"
        f"Decide on the best action to take from these options: 0, 1, 2, 3.\n"
        "Return only the action number as an integer."
    )
    
    # Call the OpenAI ChatCompletion API.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can make this configurable.
        messages=[
            {"role": "system", "content": "You are a decision-making agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Lower temperature for more deterministic output.
        max_tokens=5,     # Only a short response is expected.
    )
    
    # Extract and parse the output.
    action_text = response["choices"][0]["message"]["content"].strip()
    try:
        action = int(action_text)
    except ValueError:
        # If parsing fails, fall back to a default action (e.g., 0).
        action = 0
    return action

# Example usage:
if __name__ == "__main__":
    # Example state (here, a 10-dimensional vector)
    example_state = np.zeros((10,))
    action = agent_policy(example_state)
    print("Selected Action:", action)
