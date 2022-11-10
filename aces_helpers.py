import sys
from poker_env import PokerEnv
from agents.random_policy import RandomActions
from agents.heuristic_policy import HeuristicPolicy
from ray.rllib.algorithms.ppo import PPOConfig
from gym import spaces
import mpu
import numpy as np
import ray
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
import tensorflow as tf
import torch

heuristic_observation_space = spaces.Dict({
            "hand": spaces.Box(0, 1, shape=(24, )),
            "community": spaces.Box(0, 1, shape=(24, ))
        })
action_space = spaces.Discrete(3)

def create_env_etc(model):
    def select_policy(agent_id, episode, **kwargs):
    if agent_id == 0:
        return "learned"
    elif agent_id == 1:
        return "Heuristic_10"
    elif agent_id == 2:
        return "Heuristic_100"
    elif agent_id == 3:
        return "Heuristic_1000"
    #TODO throw exception.
    return "Heuristic_1000"

    def env_creator(config):
        env = PokerEnv(select_policy, config)
        return env

    register_env("poker", lambda config: env_creator(config))

    config = (
        PPOConfig()
        #Each rollout worker uses a single cpu
        .rollouts(num_rollout_workers=2, num_envs_per_worker=1)\
        .training(train_batch_size=4000, gamma=0.99, model=model, lr=0.0004)\
        .environment(disable_env_checking=True)\
        .multi_agent(
            policies={
                #These policies thave pre-definded polices that dont learn.
                "random": PolicySpec(policy_class=RandomActions),
                "Heuristic_10": (HeuristicPolicy, heuristic_observation_space, action_space, {'difficulty': 0}),
                "Heuristic_100": (HeuristicPolicy, heuristic_observation_space, action_space, {'difficulty': 1}),
                "Heuristic_1000": (HeuristicPolicy, heuristic_observation_space, action_space, {'difficulty': 2}),
                #Passing nothing causes this agent to deafult to using a PPO policy
                "learned": PolicySpec(
                    config={}
                ),
            },
            policy_mapping_fn=select_policy,
            policies_to_train=['learned'],
        )\
        .resources(num_gpus=0)\
        .framework('torch')
    )
    trainer = config.build(env="poker")
    return config, trainer

