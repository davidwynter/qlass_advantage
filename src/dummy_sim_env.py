class DummyEnvSimulator:
    def __init__(self):
        self.current_step = 0
        self.max_steps = 5  # Define the episode length

    def reset(self):
        """
        Resets the environment to an initial state.
        Returns:
            state (str): The initial textual state.
        """
        self.current_step = 0
        state_text = "DummyEnv: Starting state at step 0."
        return state_text

    def step(self, action):
        """
        Simulates taking an action in the environment.
        
        Args:
            action (int): The action selected by the agent.
        
        Returns:
            next_state (str): A textual description of the next state.
            reward (float): A reward computed based on the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information (empty in this dummy example).
        """
        self.current_step += 1
        # Create a new state text that includes the step number and the action taken.
        next_state = f"DummyEnv: State at step {self.current_step} after action {action}."
        
        # Define a simple reward function: even actions receive a higher reward.
        reward = 0.5 if action % 2 == 0 else 0.2
        
        # Episode ends when max_steps is reached.
        done = self.current_step >= self.max_steps
        info = {}
        return next_state, reward, done, info

# Example usage:
if __name__ == "__main__":
    env = DummyEnvSimulator()
    state = env.reset()
    print("Initial state:", state)
    
    for _ in range(6):
        # For demonstration, select a random action from 0 to 3.
        import random
        action = random.randint(0, 3)
        next_state, reward, done, info = env.step(action)
        print(f"Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break
