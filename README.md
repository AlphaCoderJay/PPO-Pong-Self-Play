# PPO-Pong-Self-Play
# PPO Self-Play Pong

I built this project to learn more about reinforcement learning by making an environment from scratch and training two PPO agents to play Pong against each other.

The two agents learn through self-play. One controls the left paddle and the other controls the right paddle, and they both receive their own observations, actions and rewards from the same game.

I also added a human-vs-agent mode so I could actually play against one of the trained agents.

## What it looks like

**AI vs AI**


https://github.com/user-attachments/assets/e5879890-14b8-4b64-a0a7-cce9f39137ff

The video length was too long so this video is in 1.6x speed



**Human vs AI**

*Add recording/GIF here*

---

## How it works

The game was made using **Pygame**, while the agents were built and trained using **PyTorch**.

Each paddle has its own PPO agent:

```text
              Pong Environment
             /                \
            /                  \
     Left Agent             Right Agent
        PPO                     PPO
         ↑                       ↑
    Observation            Observation
         ↓                       ↓
       Action                  Action
            \                  /
             \                /
              Game Step
```

Both agents play at the same time. After every step, the environment gives each agent its own reward.

I used separate actor-critic networks for the two paddles rather than sharing one network.

---

## Observation Space

Each agent receives 6 values:

```text
[
    my_paddle_y,
    opponent_paddle_y,
    ball_x,
    ball_y,
    ball_x_velocity,
    ball_y_velocity
]
```

The positions are normalised between roughly 0 and 1, while the ball velocities are normalised relative to the ball speed.

The observation is from the perspective of the individual paddle. This means the left and right agents each know where they are, where the opponent is, and where the ball is.

---

## Action Space

There are 2 possible actions:

```text
0 = UP
1 = DOWN
```

There isn't a separate "do nothing" action. If an agent chooses an action, the paddle moves in that direction for that environment step.

---

## Rewards

I kept the reward system pretty simple.

When an agent hits the ball:

```text
+1 reward
```

Missing the ball doesn't directly give a negative reward.

I originally experimented with giving rewards and penalties for scoring, but I ended up keeping the main reward focused on successfully hitting the ball.

This was one of the things I learned from my earlier RL projects: **reward design can completely change what an agent learns to do.**

---

## PPO

The agents use an Actor-Critic network.

The actor decides which action to take and the critic estimates the value of the current state.

The network is:

```text
Observation (6)
      ↓
Linear
      ↓
ReLU
      ↓
Linear
      ↓
ReLU
      ↓
Action / Value
```

The actor outputs logits for the two actions, which are turned into a categorical distribution.

I implemented the main PPO components myself, including:

* Rollout collection
* Actor-Critic networks
* GAE
* Returns
* PPO clipping
* Policy loss
* Value loss
* Entropy
* Gradient clipping
* Approximate KL tracking
* Clip fraction tracking

---

## GAE

I used Generalised Advantage Estimation to calculate the advantages used by PPO.

The main idea was to estimate how much better or worse an action turned out to be compared with what the critic expected.

I also had to deal with episode termination correctly when calculating the next value and GAE.

This was one of the parts of PPO that took me a while to properly understand.

---

## Self-Play

Both agents are trained at the same time.

During a rollout:

1. Both agents receive their observations.
2. Both agents choose an action.
3. The environment advances.
4. Each agent receives its own reward.
5. The transition is stored.
6. After the rollout, GAE and returns are calculated separately.
7. PPO updates each agent separately.

The two agents therefore learn from playing against each other rather than against a scripted opponent.

This also made training more interesting because the opponent was constantly changing as it learned.

---

## Training

The current training setup uses:

```text
Algorithm: PPO
Agents: 2
Observation size: 6
Action size: 2
Hidden size: 64
Learning rate: 3e-4
Rollout length: 2048
PPO epochs: 10
Batch size: 64
Clip epsilon: 0.2
Gamma: 0.99
GAE lambda: 0.95
```

The current run is being trained for **500 iterations**.

### Results

I'll add the final results from the 500-iteration run here.

```text
Training iterations: 500

Left agent:
Average evaluation reward: TBD

Right agent:
Average evaluation reward: TBD
```

I'll also add an AI-vs-AI recording here once the final model has finished training.

---

## Human vs Agent

After training, I made a separate test where the trained left agent plays against me.

The AI controls the left paddle and I control the right paddle using the arrow keys.

```text
AI:    Left paddle
       ↑ / ↓ chosen by PPO

Human: Right paddle
       ↑ / ↓ controlled by keyboard
```

This was a nice final test because it let me see whether the agent could actually react to something other than the exact opponent it had been training against.

*Add human-vs-agent recording here*

---

## What I learned

This project taught me a lot more about reinforcement learning than just implementing PPO.

### Reward design matters a lot

One of the biggest things I learned was how much the reward function affects the behaviour an agent learns.

I had problems in previous environments where the agent found an easy behaviour that technically gave it a reward but wasn't actually what I wanted.

Pong made this much easier to see because I could directly watch what the agents were doing.

### Self-play is harder than normal RL

With two learning agents, the environment isn't really stationary.

The opponent is also improving, which means the strategy that worked before might stop working later.

I had to think about both agents at the same time instead of only thinking about one agent learning against a fixed environment.

### I understand PPO much better now

I had already implemented PPO before this project, but building the whole thing into an actual game helped me understand what all the pieces were doing.

In particular, I got more comfortable with:

* Rollouts
* Advantages
* Returns
* GAE
* Policy ratios
* PPO clipping
* Value estimation
* Entropy
* Episode termination

### Observations need to contain useful information

I learned that giving an agent more numbers doesn't automatically make it smarter.

The observation needs to contain the information the agent actually needs to make a decision.

For Pong, the paddle positions, ball position and ball velocity were enough to give the agents the information needed to play.

### Debugging RL is different

A normal program can usually be debugged by finding where the code breaks.

With RL, the code can run perfectly while the agent does something completely stupid.

That means I had to look at things like:

* What the agent was actually seeing
* What actions it was choosing
* Whether rewards were being generated
* Whether the environment behaved correctly
* Whether GAE was calculated correctly
* Whether the PPO update was actually changing the policy

That was probably one of the most useful things I learned from this project.

---

## Why I made this

I wanted to stop just following RL tutorials and actually build environments myself.

I started with simpler RL projects and eventually wanted to try self-play. I originally experimented with a soccer environment, but I couldn't get the agents to learn useful behaviour consistently.

Instead of spending forever trying to force that environment to work, I decided to make the problem simpler and move to Pong.

That ended up being a much better environment for learning self-play because the game has a very clear objective and relatively small observation/action spaces.

---

## Project Structure

```text
Pong/
│
├── environment.py
├── agent.py
├── train.py
├── test.py
│
├── LeftPaddleTrain/
│   └── ...
│
├── RightPaddleTrain/
│   └── ...
│
└── README.md
```

---

## Running it

Install the dependencies:

```bash
pip install torch pygame numpy
```

Train the agents:

```bash
python train.py
```

The trained models are saved after training.

You can then run the evaluation/human-vs-agent script to play against the trained model.

---

## Final thoughts

This project was mainly about learning rather than trying to create the strongest Pong AI possible.

The goal was to understand how self-play works, build the environment myself, implement PPO and actually get the agents to learn a useful behaviour.

Seeing the two agents go from random movement to being able to rally the ball for a long time was probably the best part of the project.
