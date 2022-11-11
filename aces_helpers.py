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
from ray.rllib.policy.policy import Policy

from ray.tune.registry import register_env
import tensorflow as tf
import torch


heuristic_observation_space = spaces.Dict({
            "hand": spaces.Box(0, 1, shape=(24, )),
            "community": spaces.Box(0, 1, shape=(24, ))
        })

PPO_Agent_observation_space = spaces.Dict({
            "obs": spaces.Box(0, 400, shape=(24+24+16+4, )),
            "state": spaces.Box(0, 1, shape=(1, )),
        })

action_space = spaces.Discrete(3)

class TrainedPolicyAgent(Policy):

    def __init__(self,*args, **kwargs):

        super().__init__(*args, **kwargs)  
        
        self.model = torch.load(args[2]['policy_checkpoint_path']+'/model.pt')
        self.model.eval()
        
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
        #return inference(is_training=tf.constant(False), observations=obs_batch, timestep=tf.constant(-1, dtype=tf.int64))['actions_0'][0]
        dic = {'obs': torch.tensor(obs_batch.reshape(1, -1))}
        return [torch.argmax(self.model(dic, [torch.tensor(np.zeros(0))], torch.tensor(np.zeros(1)))[0]).item()], [], {}

    # def get_weights(self):
    #     return None

    # def set_weights(self, weights):
    #     return None

def get_heuristic_policy(difficulty=0):
    return (HeuristicPolicy, heuristic_observation_space, action_space, {'difficulty': difficulty})
    

def create_env_etc(env_config):
    agent_names = {}
    policies = {}
    policies_to_train = []
    model = MODEL_DEFAULTS.update({'fcnet_hiddens': [512, 512], 'fcnet_activation': 'relu'})

    for i in range(4):
        agent = env_config['agent'+str(i)]
        agent_type = agent['type']
        agent_name = ''
        if agent_type == 'Heuristic':
            policies[i] = get_heuristic_policy(agent['difficulty'])
            agent_name = 'Heuristic_'+str(i)
        elif agent_type == 'ppo':
            if 'policy_checkpoint_path' in agent:
                policy_checkpoint_path = agent['policy_checkpoint_path']
                print(policy_checkpoint_path)
                policies[i] = (TrainedPolicyAgent, PPO_Agent_observation_space, action_space,{'policy_checkpoint_path' : policy_checkpoint_path})
                #policies[i] = (TrainedPolicyAgent, PPO_Agent_observation_space, action_space, policy_checkpoint_path, {})
            else:
                print('using default policy spec')
                policies[i] = PolicySpec(config={})
            agent_name =  'ppo_'+str(i)
        if 'train' in agent:
            if agent['train'] is True:
                policies_to_train.append(agent_name)
        #TODO ensure agent_name is not blank
        agent_names[i] = agent_name

    print(policies_to_train)

    def select_policy(agent_id, episode, **kwargs):
        return agent_names[agent_id]

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
                agent_names[0]: policies[0],
                agent_names[1]: policies[1],
                agent_names[2]: policies[2],
                agent_names[3]: policies[3],
            },
            policy_mapping_fn=select_policy,
            policies_to_train=policies_to_train,
        )\
        .resources(num_gpus=0)\
        .framework('torch')
    )
    trainer = config.build(env="poker")
    return config, trainer

