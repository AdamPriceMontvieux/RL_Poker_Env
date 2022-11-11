#!/usr/bin/env python3

"""

"""

__author__ = "Daniel Craig"
__copyright__ = "Copyright 2021, Montvieux"
__version__ = "1.0.0"
__maintainer__ = "Daniel Craig"
__email__ = "daniel.craig@montvieux.com"
__status__ = "Development"

import os
import logging
import random

import torch as torch

from poker_env import PokerEnv
from ray.rllib.algorithms.ppo import PPOConfig
from gym import spaces
import numpy as np
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec
from agents.heuristic_policy import HeuristicPolicy
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

trained_policies=[]
def select_policy(agent_id, episode, **kwargs):
    return ["learned", "agent1", "agent2", "agent3"][agent_id]

def policies_to_train():
    return ["learned", "agent1", "agent2", "agent3"][0:1+(3-len(trained_policies))]

def get_policy_def():
    policies = {
        "learned": PolicySpec(
            config={}
        )
    }

    for i in range(1,4):
        agent_key = "agent" + str(i)
        if len(trained_policies) > i:
            policies[agent_key] = (classmap[agent_key], PPO_Agent_observation_space, action_space, {})
        else:
            policies[agent_key] = PolicySpec(config={})

    return policies

def env_creator(config):
    return PokerEnv(select_policy, config)

register_env("poker", lambda config: env_creator(config))

PPO_Agent_observation_space = spaces.Dict({
    "obs": spaces.Box(0, 400, shape=(24+24+16+4, )),
    "state": spaces.Box(0, 1, shape=(1, ))
})

heuristic_observation_space = spaces.Dict({
    "hand": spaces.Box(0, 1, shape=(24, )),
    "community": spaces.Box(0, 1, shape=(24, ))
})
action_space = spaces.Discrete(3)

#Defines the learning models architecture.
model = MODEL_DEFAULTS.update({'fcnet_hiddens': [512, 512], 'fcnet_activation': 'relu'})

class TrainedPolicyAgent(Policy):
    def get_initial_state(self):
        return [0]

    def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            **kwargs
    ):
        dic = {'obs': torch.tensor(obs_batch.reshape(1, -1))}
        return [torch.argmax(self.model(dic, [torch.tensor(np.zeros(0))], torch.tensor(np.zeros(1)))[0]).item()], [], {}

    def get_weights(self):
        return None

    def set_weights(self, weights):
        return None

class TrainedPolicyAgent1(TrainedPolicyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = trained_policies[0]

class TrainedPolicyAgent2(TrainedPolicyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = trained_policies[1]

class TrainedPolicyAgent3(TrainedPolicyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = trained_policies[2]

classmap = {
    "agent1": TrainedPolicyAgent1,
    "agent2": TrainedPolicyAgent2,
    "agent3": TrainedPolicyAgent3,
}

print("Starting Training")
for run in range(1):
    config = (
        PPOConfig()
        # Each rollout worker uses a single cpu
        .rollouts(num_rollout_workers=2, num_envs_per_worker=1) \
        .training(train_batch_size=4000, gamma=0.99, model=model, lr=0.0004) \
        .environment(disable_env_checking=True) \
        .multi_agent(
            policies=get_policy_def(),
            policy_mapping_fn=select_policy,
            policies_to_train=policies_to_train(),
        ) \
        .resources(num_gpus=0) \
        .framework('torch')
    )
    trainer = config.build(env="poker")

    iterations = 10
    for i in range(iterations):
        print("Run %d Iteration %d with policy def %s" % (run, i, get_policy_def()))
        trainer.train()

    learned_policy = trainer.get_policy('learned')
    print("learned_policy %s" % learned_policy.__dict__)

    trained_policies[random.randint(0, 3)] = learned_policy


    iterations = round(iterations * 1.5)