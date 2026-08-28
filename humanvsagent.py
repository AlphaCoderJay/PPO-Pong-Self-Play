import random
import pygame
import torch
from environment import PingPongEnv
from agent import ActorCritic


env = PingPongEnv(render=True)

done = False
state = env.reset()
left_agent = ActorCritic()


left_agent.load_state_dict(
    torch.load("LeftPaddleTrain/left500.pth")
)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            raise SystemExit

    keys = pygame.key.get_pressed()

    left_paddle_agent_obs = env.get_player_observation("left_paddle_agent")
    left_obs_tensor = torch.FloatTensor(left_paddle_agent_obs)
    with torch.no_grad():
        left_action, _, _ = left_agent.get_action(
            left_obs_tensor,
            determinisitc=False
        )

    if keys[pygame.K_UP]:
        right_action = 0
    elif keys[pygame.K_DOWN]:
        right_action = 1
    else:
        right_action = None



    left_reward,right_reward,done = env.step(
        left_action,
        right_action
    )



