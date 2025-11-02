from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

my_agent = SubmittedAgent(file_path='checkpoints/experiment_12/rl_model_11643036_steps.zip')

#Input your file path here in SubmittedAgent if you are loading a model:
# opponent = SubmittedAgent(file_path=None)
# opponent = SB3Agent(file_path='checkpoints/experiment_9/rl_model_336052_steps.zip')
opponent = UserInputAgent()
match_time = 99999

# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)