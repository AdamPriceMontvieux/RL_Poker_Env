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
import sys
import uuid
import logging

import torch as torch

from poker_env import PokerEnv
from agents.random_policy import RandomActions
from agents.heuristic_policy import HeuristicPolicy
from ray.rllib.algorithms.ppo import PPOConfig
from gym import spaces
import mpu
import numpy as np
import ray
import tqdm
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

trained_policies=[]
def select_policy(agent_id, episode, **kwargs):
    return ["learned1", "learned2", "learned3", "learned4"][agent_id]

def policies_to_train():
    return ["learned", "agent1", "agent2", "agent3"][0:1+(3-len(trained_policies))]

def get_policy_def():
    policies = {
        "learned": PolicySpec(
            config={}
        )
    }

    for i in range(1,4):
        if len(trained_policies) > i:
            policies["agent" + i] = (classmap["agent" + i], PPO_Agent_observation_space, action_space, {})
        else:
            policies["agent" + i] = PolicySpec(
                config={}
            )

    return policies

def env_creator(config):
    env = PokerEnv(select_policy, config)
    return env

register_env("poker", lambda config: env_creator(config))

PPO_Agent_observation_space = spaces.Dict({
            "obs": spaces.Box(0, 400, shape=(24+24+16+4, )),
            "state": spaces.Box(0, 1, shape=(1, )),
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
        # return inference(is_training=tf.constant(False), observations=obs_batch, timestep=tf.constant(-1, dtype=tf.int64))['actions_0'][0]
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

for run in range(10):
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

    iterations = 20
    for i in range(iterations):
        logger.debug("Run %d Iteration %d" % (run, i))
        trainer.train()

    learned_policy = trainer.get_policy('learned')
    logger.debug("learned_policy %s" % learned_policy.__dict__)


    iterations *= 1.5