import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self,obs_dim=6,act_dim=2, hidden=64):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )


    def forward(self,x):
        logits = self.actor(x)
        values = self.critic(x)

        return logits,values.squeeze(-1)

    def get_action(self,obs,determinisitc=False):
        logits,values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)

        if determinisitc:
            action = logits.argmax(dim=-1)

        else:
            action = dist.sample()


        log_prob = dist.log_prob(action)
        return action,log_prob,values



def collect_self_play_rollout(left_paddle_agent, right_paddle_agent, env, num_steps=5000):
    left_paddle_agent_transitions = []
    right_paddle_agent_transitions = []

    env.reset()

    left_paddle_agent_obs = env.get_player_observation("left_paddle_agent")
    right_paddle_agent_obs = env.get_player_observation("right_paddle_agent")


    for _ in range(num_steps):
        left_paddle_agent_obs = torch.FloatTensor(left_paddle_agent_obs)
        right_paddle_agent_obs = torch.FloatTensor(right_paddle_agent_obs)

        with torch.no_grad():
            left_paddle_agent_action,left_paddle_agent_log_prob, left_paddle_agent_value = left_paddle_agent.get_action(left_paddle_agent_obs)
            right_paddle_agent_action,right_paddle_agent_log_prob, right_paddle_agent_value = right_paddle_agent.get_action(right_paddle_agent_obs)

        left_paddle_agent_reward, right_paddle_agent_reward, done = env.step(
            left_paddle_agent_action.item(),
            right_paddle_agent_action.item()
        )

        left_paddle_agent_transitions.append({
            "obs":left_paddle_agent_obs,
            "action": left_paddle_agent_action.item(),
            "log_prob": left_paddle_agent_log_prob.item(),
            "value": left_paddle_agent_value.item(),
            "reward": float(left_paddle_agent_reward),
            "terminated": done
        })

        right_paddle_agent_transitions.append({
            "obs":right_paddle_agent_obs,
            "action":right_paddle_agent_action.item(),
            "log_prob": right_paddle_agent_log_prob.item(),
            "value": right_paddle_agent_value.item(),
            "reward" : float(right_paddle_agent_reward),
            "terminated": done
        })


        if done:
            env.reset()
            left_paddle_agent_obs = env.get_player_observation("left_paddle_agent")
            right_paddle_agent_obs = env.get_player_observation("right_paddle_agent")

        else:
            left_paddle_agent_obs = env.get_player_observation("left_paddle_agent")
            right_paddle_agent_obs = env.get_player_observation("right_paddle_agent")


    
    return left_paddle_agent_transitions,right_paddle_agent_transitions

def compute_gae(transitions,gamma=0.99,lam=0.95):
    n = len(transitions)

    rewards = [t["reward"] for t in transitions]
    values = [t["value"] for t in transitions]
    terminated = [t["terminated"] for t in transitions]


    advantage = [0.0] * n
    gae = 0.0

    for t in reversed(range(n)):
        if t == n -1:
            next_value = 0

        else:
            next_value = values[t+1]


        if terminated[t]:
            next_value = 0.0


        delta = rewards[t] + gamma * next_value - values[t]

        gae = delta + gamma * lam * (0 if terminated[t] else gae)

        advantage[t] = gae

    advantage = torch.FloatTensor(advantage)
    returns = advantage + torch.FloatTensor(values)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    return advantage,returns



def ppo_update(agent,optimizer,transitions,advantages,returns,clip_eps= 0.2, epochs = 10,batch_size=64):
    obs = np.array([t["obs"] for t in transitions])
    actions = np.array([t["action"] for t in transitions])
    old_log_prob = np.array([t["log_prob"] for t in transitions])

    obs = torch.FloatTensor(obs)
    actions = torch.LongTensor(actions)
    old_log_prob = torch.FloatTensor(old_log_prob)

    total_policy_loss = 0
    total_entrophy = 0
    total_value_loss = 0

    total_kl = 0
    total_clip_frac = 0
    n_updates = 0

    for _ in range(epochs):
        indices = np.random.permutation(len(transitions))

        for start in range(0,len(transitions), batch_size):
            idx = indices[start: start + batch_size]

            batch_obs = obs[idx]
            batch_action = actions[idx]
            batch_old_log_prob = old_log_prob[idx]
            batch_advantage = advantages[idx]
            batch_returns = returns[idx]

            logits,values = agent(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = dist.log_prob(batch_action)

            ratio = torch.exp(new_log_prob - batch_old_log_prob)
            surr1 = ratio * batch_advantage
            surr2 = torch.clamp(surr1,1 - clip_eps, 1 + clip_eps) * batch_advantage
            policy_loss = -torch.min(surr1,surr2).mean()

            value_loss = ((values - batch_returns) ** 2).mean()

            entrophy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entrophy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(),0.5)
            optimizer.step()


            with torch.no_grad():
                total_kl += (batch_old_log_prob - new_log_prob).mean().item()
                total_clip_frac += ((ratio - 1.0).abs() > clip_eps).float().mean().item()



            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entrophy += entrophy.item()
            n_updates +=1


    return{
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entrophy / n_updates,
        "approx_kl": total_kl / n_updates,
        "clip_fraction": total_clip_frac / n_updates,
    }