#!/usr/bin/env python3

"""

"""

__author__ = "Daniel Craig"
__copyright__ = "Copyright 2021, Montvieux"
__version__ = "1.0.0"
__maintainer__ = "Daniel Craig"
__email__ = "daniel.craig@montvieux.com"
__status__ = "Development"

import copy
import json
import os
import logging
import shutil

import torch as torch

from poker_env import PokerEnv
from ray.rllib.algorithms.ppo import PPOConfig
from agents.heuristic_policy import HeuristicPolicy
from gym import spaces
import numpy as np
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

trained_policies=[]
def select_policy(agent_id, episode, **kwargs):
    if len(trained_policies) == 0:
        return ["learned", "Heuristic_100_1", "Heuristic_100_2", "Heuristic_100_3"][agent_id]
    else:
        return ["learned", "agent1", "agent2", "agent3"][agent_id]

def policies_to_train():
    policies = ["learned"]

    return policies

def get_policy_def():
    dqn_config = DQN.get_default_config()
    dqn_config.update({
        "n_step": 1, "noisy": False, "num_atoms": 31, "v_min": -50.0, "v_max": 50.0
    })
    policies = {
        "learned": PolicySpec(config=dqn_config)
    }

    for i in range(1,4):
        agent_key = "agent" + str(i)
        print("Getting policy for %s" % (agent_key))
        if len(trained_policies) > 0:
            print("loading %s" % classmap[agent_key].__name__)
            policies[agent_key] = (classmap[agent_key], PPO_Agent_observation_space, action_space, {})
        else:
            policies["Heuristic_100_" + str(i)] = (HeuristicPolicy, heuristic_observation_space, action_space, {'difficulty': 1})

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.model = trained_policies[0]["weights"]

class TrainedPolicyAgent2(TrainedPolicyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = trained_policies[1]["weights"]

class TrainedPolicyAgent3(TrainedPolicyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = trained_policies[2]["weights"]

classmap = {
    "agent1": TrainedPolicyAgent1,
    "agent2": TrainedPolicyAgent2,
    "agent3": TrainedPolicyAgent3,
}

def evaluate(policy, opponents):
    policy_return = 0
    policies = [policy]
    policies.extend(opponents)

    def select_agent_policy(agent_id, episode, **kwargs):
        return ["learned", "agent1", "agent2", "agent3"][agent_id]

    env = PokerEnv(select_agent_policy, {})
    for i in range(100):
        state = env.reset()[env.current_actor]
        done = False
        while done == False:
            s = np.array([np.concatenate([state['obs'], state['state']])])
            action = policies[env.current_actor].compute_actions(s, training=False)
            state_g = env.step({env.current_actor: action[0][0]})
            done = state_g[2]['__all__']
            state = state_g[0][env.current_actor]

        policy_return += state_g[1][0]

    return policy_return

trainer = None
iterations = 200

print("Starting Training")
for run in range(999):
    reward_means = []
    policy_def = get_policy_def()
    train_policies = policies_to_train()
    config = PPOConfig() \
        .rollouts(num_rollout_workers=8, num_envs_per_worker=1, preprocessor_pref="deepmind") \
        .training(train_batch_size=10000, gamma=0.99, model=model, lr=0.0004) \
        .environment(disable_env_checking=True) \
        .multi_agent(
            policies=policy_def,
            policy_mapping_fn=select_policy,
            policies_to_train=train_policies,
        ) \
        .resources(num_gpus=0) \
        .framework('torch')

    # create or update the trainer
    if trainer is not None:
        trainer.reset_config(config)
    else:
        trainer = config.build(env="poker")

    for i in range(iterations):
        print("Run %d Iteration %d" % (run, i))
        batch = trainer.train()
        # reward_means = (reward_means + [batch["policy_reward_mean"]["learned"]])[-20:]
        #
        # print(reward_means)
        #
        # if i > 20:
        #     print((sum(reward_means)/20), (sum(reward_means[-10:])/10))
        #     if (sum(reward_means)/20) > (sum(reward_means[-10:])/10):
        #         break

    learned_policy = trainer.get_policy('learned')

    # append of inject the new policy in as an opponent for next round
    if len(trained_policies) == 0:
        # evaluate against the other dumb opponents from this round
        evaluated_return = evaluate(learned_policy, [
            learned_policy,
            learned_policy,
            learned_policy
        ])

        for i in range(3):
            trained_policies.append({
                "policy": learned_policy,
                "weights": copy.deepcopy(learned_policy.get_weights()),
                "return": evaluated_return
            })

        print("Learned policy return is %d, replacing all opponents" % evaluated_return)
    else:
        # evaluate against the other opponents from other rounds
        evaluated_return = evaluate(learned_policy, [
            trained_policies[0]["policy"],
            trained_policies[1]["policy"],
            trained_policies[2]["policy"]
        ])
        print("Learned policy return is %d, replacing inferior opponents" % evaluated_return)
        # loop through and replace all inferior policies
        for i, p in enumerate(trained_policies):
            print("agent%d is %d" % (i, p["return"]))
            if p["return"] < evaluated_return:
                print("replacing inferior agent%d" % (i))
                p["policy"] = learned_policy
                p["weights"] = copy.deepcopy(learned_policy.get_weights())
                p["return"] = evaluated_return

                # export model for submission
                model_path = 'models/dc_ppo_agent'
                if os.path.isdir(model_path):
                    shutil.rmtree(model_path)
                os.makedirs(model_path, )

                print("Saving model to %s" % model_path)
                learned_policy.export_model(model_path)

                break

    # iterations = round(iterations * 1.5)