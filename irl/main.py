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
from emgail import EMGail
from mlp_model import EMGPolicy, EMGValue, GPolicy
from samples_dataset import SamplesDataset




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
parser.add_argument('--max-timesteps', default=500, type=int)
parser.add_argument('--n-intentions', default=2, type=int)


# IRL options
parser.add_argument('--irl-type', default='EMGAIL', type=str)

# RL options
parser.add_argument('--rl-type', default='TRPO', type=str)
parser.add_argument('--num-sample-trajs', default=5, type=int)
parser.add_argument('--rl-maxiter', default=500, type=int)
parser.add_argument('--criteria', default=0.8, type=int)


# Demonstrations
parser.add_argument('--expert-max-timesteps', default=500, type=int)
parser.add_argument('--test-max-timesteps', default=300, type=int)
parser.add_argument('--n-demons', default=5, type=int)

# Output
parser.add_argument('--output-folder', default='checkpoints')
parser.add_argument('--exp-name', default='exp1')

# Control parameters
parser.add_argument('--train-irl', default=False, type=bool)
parser.add_argument('--test-irl', default=False, type=bool)
parser.add_argument('--test-intention', default=0, type=int)



def main(args):
    args.max_timesteps = args.expert_max_timesteps
    if args.test_irl:
        args.max_timesteps = args.test_max_timesteps
    

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    env = gym.make(args.env_name)
    env.seed(args.seed)
    env._max_episode_steps = args.max_timesteps
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.intention_dim = args.n_intentions

    random_policy = GPolicy(args.intention_dim, args.state_dim, args.action_dim)
    value_fun = EMGValue(args.intention_dim, args.state_dim)
    policy = EMGPolicy(args.intention_dim, args.state_dim, args.action_dim)
    irl_net = EMGail(args.intention_dim, args.state_dim, args.action_dim, device)

    policy.to(device)
    value_fun.to(device)
    random_policy.to(device)
    

    all_demons = []
    for intention_idx in range(args.n_intentions):
        demons_path = os.path.join(
            'irl/',args.output_folder,args.env_name+'_'+'Demons'+'_'+str(intention_idx)+'.pth'
        )
        demons = torch.load(demons_path)
        all_demons.append(demons)
    
    expert_dataset = SamplesDataset(all_demons, args.n_intentions)
    

    z_filter = ZFilter()
    state_bound = Bound(-5, 5)
    state_filter = Transform(state_bound, z_filter)

    sampler = Sampler(
        args.env_name, policy, args.num_sample_trajs, args.max_timesteps, 
        device, state_filter=state_filter, seed=args.seed
    )

    rl_net = TRPO(policy, value_fun, sampler, device)
    
    agent = Agent(rl_net, sampler, policy, env, device, irl_net, expert_dataset, args.irl_type, random_policy, args.n_intentions)
    max_AERD, min_AERD = agent.test_random_policy()


    if args.test_irl:
        logger.info('Testing IRL policy ...')

        checkpoint_path = os.path.join(
            'irl/',args.output_folder,args.env_name+'_'+str(args.n_intentions)+'_'+args.irl_type+'_'+args.exp_name+'.pth'
        )
        logger.info('Restoring IRL policy from {}'.format(checkpoint_path))

        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint[0])
        agent.sampler.state_filter = checkpoint[1]
        intention_idx = args.test_intention
        intention = np.zeros(args.intention_dim)
        intention[intention_idx] = 1.
        img_path = os.path.join(
            './',args.output_folder,'img',str(intention_idx),args.env_name+'_'+str(intention_idx)
        )
        log = agent.test_policy(intention, img_path, save=False)
        print(
            'avg length: {:04d} \t avg reward:  '.format(
                int(log['avg_length'])), log['avg_reward']
            
        )

    elif args.train_irl:
        logger.info('Training IRL policy ...')

        for iteration in range(args.rl_maxiter):
            iter_start_time = time.time()
            all_logs = []
            all_samples = []
            for intention_idx in range(args.n_intentions):
                intention = np.zeros(args.intention_dim)
                intention[intention_idx] = 1.
                log, samples = agent.collect_samples(intention, intention_idx)
                all_logs.append(log)
                all_samples.append(samples)


            discrim_loss, posterior_loss, cluster_accuracy = agent.update(all_demons, all_samples)

            rewards = []
            for intention_idx in range(args.n_intentions):
                intention = np.zeros(args.intention_dim)
                intention[intention_idx] = 1.
                eval_log, _ = agent.collect_samples(intention, intention_idx, deterministic=True, n=5, update_run_state=False)
                log = all_logs[intention_idx]
                rewards.append(eval_log['avg_reward'])
                print('intention: {:01d} \t avg length: {:04d}'.format(
                    intention_idx, int(eval_log['avg_length'])
                ), '\t avg reward',eval_log['avg_reward'])

            iter_time = time.time() - iter_start_time
            AERD, correct_idx = compute_AERD(rewards, expert_dataset.rewards_true)
            AERD = normalaize_AERD(AERD, max_AERD, min_AERD)
            logger.info(
                'Iteration: {:04d} \t time: {:.3f} \t AERD {:.3f}\t discriminator loss {:.3f}\t posterior loss {:.3f}\t accuracy {:.3f}\t criteria {:.3f}'.format(
                    iteration, iter_time, AERD, discrim_loss, posterior_loss, cluster_accuracy, args.criteria
                )
            )

            if AERD < args.criteria:
                args.criteria = AERD
                logger.info('Criteria is updated to {:.4f}'.format(AERD))
                checkpoint_path = os.path.join(
                    'irl/',args.output_folder,args.env_name+'_'+str(args.n_intentions)+'_'+args.irl_type+'_'+args.exp_name+'.pth'
                )
                logger.info('Storing IRL policy in {}'.format(checkpoint_path))
                checkpoint = [policy.state_dict(), state_filter, AERD, correct_idx, cluster_accuracy]
                torch.save(checkpoint, checkpoint_path)



            torch.cuda.empty_cache()
    
    


    







if __name__ == '__main__':
    args = parser.parse_args()
    main(args)