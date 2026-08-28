import torch
import pygame

from environment import PingPongEnv
from agent import ActorCritic


pygame.init()

env = PingPongEnv(render=True)

left_agent = ActorCritic()
right_agent = ActorCritic()

left_agent.load_state_dict(
    torch.load("LeftPaddleTrain/left500.pth")
)

right_agent.load_state_dict(
    torch.load("RightPaddleTrain/right500.pth")
)

left_agent.eval()
right_agent.eval()

left_scores = 0
right_scores = 0




for episode in range(5):

    env.reset() 

    done = False


    while not done:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit



        left_paddle_agent_obs = env.get_player_observation("left_paddle_agent")
        right_paddle_agent_obs = env.get_player_observation("right_paddle_agent")
        left_obs_tensor = torch.FloatTensor(left_paddle_agent_obs)
        right_obs_tensor = torch.FloatTensor(right_paddle_agent_obs)
        with torch.no_grad():

            left_action, _, _ = left_agent.get_action(
                left_obs_tensor,
                determinisitc=False
            )

            right_action, _, _ = right_agent.get_action(
                right_obs_tensor,
                determinisitc=False
            )

        left_reward, right_reward, done = env.step(
            left_action,
            right_action
        )




pygame.quit()