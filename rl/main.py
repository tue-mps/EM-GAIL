import argparse
import os
import logging
import sys
import time
import numpy as np
import torch
import gym
from utils import *
from transform import Transform, ZFilter, Bound
from sampler import Sampler
from trpo import TRPO
from agent import Agent
from mlp_policy import RLPolicy, RLValue


parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Env options
parser.add_argument('--env-name', default="Swimmer-v2", type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--max-timesteps', default=2000, type=int)
parser.add_argument('--intention-idx', default=0, type=int)

# RL options
parser.add_argument('--rl-type', default='TRPO', type=str)
parser.add_argument('--num-sample-trajs', default=10, type=int)
parser.add_argument('--rl-maxiter', default=1000, type=int)
parser.add_argument('--criteria', default=50, type=int)


# Demonstrations
parser.add_argument('--expert-max-timesteps', default=500, type=int)
parser.add_argument('--n-demons', default=5, type=int)

# Output
parser.add_argument('--output-folder', default='checkpoints')

# Control parameters
parser.add_argument('--train-rl', default=False, type=bool)
parser.add_argument('--test-rl', default=False, type=bool)
parser.add_argument('--generate-demons', default=False, type=bool)



def main(args):
    if args.generate_demons:
        args.max_timesteps = args.expert_max_timesteps

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    env = gym.make(args.env_name)

    env.seed(args.seed)
    env._max_episode_steps = args.max_timesteps
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    value_fun = RLValue(args.state_dim)
    policy = RLPolicy(args.state_dim, args.action_dim)
    policy.to(device)
    value_fun.to(device)
    

    z_filter = ZFilter()
    state_bound = Bound(-5, 5)
    state_filter = Transform(state_bound, z_filter)

    sampler = Sampler(
        args.env_name, policy, args.num_sample_trajs, args.max_timesteps, 
        device, state_filter=state_filter, seed=args.seed
    )

    trpo = TRPO(policy, value_fun, sampler, device)

    agent = Agent(trpo, sampler, policy, env, args.intention_idx, device)

    

    if args.test_rl:
        logger.info('Testing RL policy ...')

        checkpoint_path = os.path.join(
            'rl/',args.output_folder,args.env_name+'_'+args.rl_type+'_'+str(args.intention_idx)+'.pth'
        )
        logger.info('Restoring RL policy from {}'.format(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint[0])
        agent.sampler.state_filter = checkpoint[1]
        log = agent.test_policy()
        logger.info(
            'avg length: {:04d} \t avg reward: {:.3f}'.format(
                int(log['avg_length']), log['avg_reward']
            )
        )

    elif args.generate_demons:
        logger.info('Generating demonstrations ...')
        checkpoint_path = os.path.join(
            'rl/',args.output_folder,args.env_name+'_'+args.rl_type+'_'+str(args.intention_idx)+'.pth'
        )
        logger.info('Restoring RL policy from {}'.format(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint[0])
        agent.sampler.state_filter = checkpoint[1]
        demons_log, demons = agent.collect_samples(deterministic=True, n=args.n_demons, update_run_state=False)
        logger.info(
            'avg length: {:04d} \t avg reward: {:.3f}'.format(
                int(demons_log['avg_length']), demons_log['avg_reward']
            )
        )
        checkpoint_path = os.path.join(
            'rl/',args.output_folder,args.env_name+'_'+'Demons'+'_'+str(args.intention_idx)+'.pth'
        )
        logger.info('Storing demonstrations in {}'.format(checkpoint_path))
        checkpoint = demons
        torch.save(checkpoint, checkpoint_path)
        

    elif args.train_rl:
        logger.info('Training RL policy ...')

        for iteration in range(args.rl_maxiter):
            iter_start_time = time.time()

            log, samples = agent.collect_samples()
            agent.update(samples)
            
            eval_log, _ = agent.collect_samples(deterministic=True, n=5, update_run_state=False)

            iter_time = time.time() - iter_start_time

            logger.info(
                'Iteration: {:04d} \t avg length: {:04d} \t avg reward: {:.3f}  \t avg train length: {:04d}  \t avg train reward: {:.3f} \t time: {:.3f} \t criteria: {:.3f}'.format(
                    iteration, int(eval_log['avg_length']), eval_log['avg_reward'], int(log['avg_length']), log['avg_reward'], iter_time, args.criteria
                )
            )

            if eval_log['avg_reward'] > args.criteria:
                args.criteria = eval_log['avg_reward']
                logger.info('Criteria is updated to {:.4f}'.format(eval_log['avg_reward']))
                checkpoint_path = os.path.join(
                    'rl/',args.output_folder,args.env_name+'_'+args.rl_type+'_'+str(args.intention_idx)+'.pth'
                )
                logger.info('Storing RL policy in {}'.format(checkpoint_path))
                checkpoint = [policy.state_dict(), state_filter]
                torch.save(checkpoint, checkpoint_path)



            torch.cuda.empty_cache()
    
    


    







if __name__ == '__main__':
    args = parser.parse_args()
    main(args)