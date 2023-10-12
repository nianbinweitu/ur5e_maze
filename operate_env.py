import gym
import torch
import pybullet
import os
import pickle
import datetime
import numpy as np
from typing import Callable
from time import sleep, time
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc
from imitation.data import rollout
from ur5e_maze_ImageObsEnv import ImageObsUR5Sim
from ur5e_maze_MatrixObsEnv import MatrixPuzzleObsUR5Sim
from ur5e_maze_FlattenMatrixObsEnv import FlattenMatrixObsUR5Sim
from ur5e_maze_DiscreteActPuzzleEnv import DiscreteActPuzzle

gym.register(
    id="maze-v1",
    entry_point="ur5e_maze_FlattenMatrixObsEnv:FlattenMatrixObsUR5Sim",
)


class OperateRobot:
    def __init__(self, env):
        self.env = env
        now = datetime.datetime.now().strftime("%y-%m-%d/%H:%M:%S/")
        # get_path
        if isinstance(self.env, DiscreteActPuzzle):
            self.rl_path = os.path.join('ur5e_maze_result/rl/DiscreteActPuzzle', now)
            self.il_path = 'ur5e_maze_result/il/DiscreteActPuzzle'
            self.rollouts_path = 'ur5e_maze_rollouts/DiscreteActPuzzle/'
            self.rlinputpolicy = "MlpPolicy"
        elif isinstance(self.env, FlattenMatrixObsUR5Sim):
            self.rl_path = os.path.join('ur5e_maze_result/rl/FlattenMatrixObsEnv', now)
            self.il_path = 'ur5e_maze_result/il/FlattenMatrixObsEnv'
            self.rollouts_path = 'ur5e_maze_rollouts/FlattenMatrixObsEnv/'
            self.rlinputpolicy = "MlpPolicy"
        elif isinstance(self.env, MatrixPuzzleObsUR5Sim):
            self.rl_path = os.path.join('ur5e_maze_result/rl/MatrixObsEnv', now)
            self.il_path = 'ur5e_maze_result/il/MatrixObsEnv'
            self.rollouts_path = 'ur5e_maze_rollouts/MatrixObsEnv/'
            self.rlinputpolicy = "MultiInputPolicy"
        elif isinstance(self.env, ImageObsUR5Sim):
            self.rl_path = os.path.join('ur5e_maze_result/rl/ImageObsEnv', now)
            self.il_path = 'ur5e_maze_result/il/ImageObsEnv'
            self.rollouts_path = 'ur5e_maze_rollouts/ImageObsEnv/'
            self.rlinputpolicy = "CnnPolicy"
        else:
            raise AssertionError("请选择“ImagePuzzleObsUR5Sim” 或 “MatrixPuzzleObsUR5Sim“ 训练环境")

        '''rl_model'''
        # rl_train
        self.eval_path = os.path.join(self.rl_path, 'eval')
        self.tensorboard_path = self.rl_path
        self.eventualmodel_path = os.path.join(self.rl_path, 'eventual_model.zip')

        '''il_model'''
        self.rng = np.random.default_rng(0)

        # bc训练
        self.n_rollouts_load = 100
        self.n_bc_train_epochs = 100
        self.bc_policy_path = os.path.join(self.il_path, '{}rollouts'.format(self.n_rollouts_load))
        self.bc_policy_store_path = os.path.join(self.bc_policy_path, '{}epoch_bc_policy'.format(self.n_bc_train_epochs))

        # bc后的rl训练
        self.n_rl_train_steps = 1_000_000
        self.pretrained_model_path = self.bc_policy_store_path
        self.il_rl_path = os.path.join(self.bc_policy_path, 'rl_after_{}epochBC'.format(self.n_bc_train_epochs), now)

        # pretrain和train的ActorCriticPolicy载体，PPO或A2C
        self.rl_model = PPO(
            self.rlinputpolicy,
            self.env,
            policy_kwargs=dict(activation_fn=torch.nn.Tanh,
                               net_arch=dict(vf=[512, 512], pi=[512, 512])),
            learning_rate=self.linear_schedule(2e-6),
            verbose=1,
            tensorboard_log=self.il_rl_path)

        # self.expert = self.rl_model  # fake expert

    '''rl'''

    def linear_schedule(self, initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func

    def rl_train(self, rl_algorithm):
        if self.env.method == pybullet.GUI:
            raise AssertionError("训练pybullet环境任务，请将env.method设置为pybullet.DIRECT")
        env = DummyVecEnv([lambda: self.env])
        # 对环境的观测值进行归一化处理
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                           clip_obs=10.)
        if rl_algorithm == "sac":
            model = SAC(
                self.rlinputpolicy,
                env,
                # policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                #                    net_arch=dict(qf=[128, 128], pi=[128, 128])),
                # learning_rate=0.001,  # self.linear_schedule(0.001),
                verbose=1,
                tensorboard_log=self.tensorboard_path)
        elif rl_algorithm == "ppo":
            model = PPO(
                self.rlinputpolicy,
                env,
                # policy_kwargs=dict(activation_fn=torch.nn.LeakyReLU,
                #                    net_arch=dict(vf=[128, 128], pi=[128, 128])),
                # learning_rate=self.linear_schedule(1e-4),
                verbose=1,
                tensorboard_log=self.tensorboard_path)
        else:
            raise AssertionError("{}不在选择范围内，请选择ppo或sac".format(rl_algorithm))

        time_start = time()
        eval_callback = EvalCallback(env,
                                     best_model_save_path=self.eval_path,
                                     log_path=self.eval_path,
                                     eval_freq=5000,
                                     deterministic=True,
                                     render=False)
        model.learn(total_timesteps=5_000_000, callback=eval_callback)
        time_end = time()
        print('time cost', time_end - time_start, 's')
        model.save(self.eventualmodel_path)

    def venv_rl_train(self, rl_algorithm):
        def make_env(env_id, rank, seed=0):
            """
            Utility function for multiprocessed env.

            :param env_id: (str) the environment ID
            :param num_env: (int) the number of environments you wish to have in subprocesses
            :param seed: (int) the inital seed for RNG
            :param rank: (int) index of the subprocess
            """
            def _init():
                env = gym.make(env_id)
                env.seed(seed + rank)
                return env
            set_random_seed(seed)
            return _init

        env_id = "maze-v1"
        num_cpu = 8  # Number of processes to use
        # # Create the vectorized environment
        env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

        # Stable Baselines provides you with make_vec_env() helper
        # which does exactly the previous steps for you.
        # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
        # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

        if rl_algorithm == "sac":
            model = SAC(
                self.rlinputpolicy,
                env,
                policy_kwargs=dict(activation_fn=torch.nn.LeakyReLU,
                                   net_arch=dict(qf=[128, 128], pi=[128, 128])),
                learning_rate=self.linear_schedule(5e-4),
                verbose=1,
                tensorboard_log=self.tensorboard_path)
        elif rl_algorithm == "ppo":
            model = PPO(
                self.rlinputpolicy,
                env,
                # policy_kwargs=dict(activation_fn=torch.nn.LeakyReLU,
                #                    net_arch=dict(vf=[128, 128], pi=[128, 128])),
                # learning_rate=self.linear_schedule(1e-4),
                verbose=1,
                tensorboard_log=self.tensorboard_path)
        else:
            raise AssertionError("{}不在选择范围内，请选择ppo或sac".format(rl_algorithm))

        time_start = time()
        eval_callback = EvalCallback(env,
                                     best_model_save_path=self.eval_path,
                                     log_path=self.eval_path,
                                     eval_freq=5000,
                                     deterministic=True,
                                     render=False)
        model.learn(total_timesteps=5_000_000, callback=eval_callback)
        time_end = time()
        print('time cost', time_end - time_start, 's')
        model.save(self.eventualmodel_path)

    def test_rl_model(self, model_type, model_path, n_episode=0):
        test_env = self.env
        if model_type == "sac":
            model = SAC.load(model_path, env=test_env)
        elif model_type == "ppo":
            model = PPO.load(model_path, env=test_env)
        elif model_type == "a2c":
            model = A2C.load(model_path, env=test_env)
        else:
            raise AssertionError("{}不在选择范围内，请选择ppo/sac/a2c".format(model_type))
        print("正在评估 rl_policy ......")
        reward, _ = evaluate_policy(model, test_env, 20)
        print("model_20episodes_average_reward:", reward)

        if test_env.method != pybullet.GUI:
            raise AssertionError("pybullet仿真环境不可视，请将env.method设置为pybullet.GUI")
        obs = test_env.reset()
        episode_reward = 0
        for _ in range(n_episode):
            done = False
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                episode_reward += reward
                sleep(0.1)
            print("episode_reward =", episode_reward)
            obs = test_env.reset()
            episode_reward = 0

    def rl_retrain(self, model_path, store_path, rl_algorithm):
        if self.env.method == pybullet.GUI:
            raise AssertionError("训练pybullet环境任务，请将env.method设置为pybullet.DIRECT")
        env = DummyVecEnv([lambda: self.env])
        if rl_algorithm == "sac":
            model = SAC.load(model_path, learning_rate=4e-4, env=env)
        elif rl_algorithm == "ppo":
            model = PPO.load(model_path, env=env)
        else:
            raise AssertionError("{}不在选择范围内，请选择ppo或sac".format(rl_algorithm))
        time_start = time()
        eval_callback = EvalCallback(env,
                                     best_model_save_path=os.path.join(store_path, 'eval_'),
                                     log_path=os.path.join(store_path, 'eval_'),
                                     eval_freq=100,
                                     deterministic=True,
                                     render=False)
        model.learn(total_timesteps=2_000_000, reset_num_timesteps=False, callback=eval_callback)
        time_end = time()
        print('time cost', time_end - time_start, 's')
        model.save(os.path.join(store_path, 'eventual_model_.zip'))

    '''il'''

    def generate_rollouts(self, n_rollouts_generate):
        expert = None
        # 生成rollouts
        rollouts_store_path = os.path.join(self.rollouts_path, 'expert_rollouts{}.pkl'.format(n_rollouts_generate))
        env = self.env
        # 生成rollouts
        print("rollout start")
        rollouts = rollout.rollout(
            expert,
            DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
            rollout.make_sample_until(min_episodes=n_rollouts_generate),
            rng=self.rng,
        )
        print("rollout over")
        print(rollouts)
        # 储存rollouts
        with open(rollouts_store_path, 'wb') as f:
            pickle.dump(rollouts, f)

    def rollouts_to_transitions(self, rollouts_load_path):
        print("正在打开 rollouts 文件 ......")
        with open(rollouts_load_path, 'rb') as f:
            rollouts = pickle.load(f)
        print("正在转化为 transitions ......")
        transitions = rollout.flatten_trajectories(rollouts)
        print("transitions 转化完成！")
        return transitions

    def bc_pretrain(self, transitions):
        if self.env.method == pybullet.GUI:
            raise AssertionError("训练pybullet环境任务，请将env.method设置为pybullet.DIRECT")
        env = DummyVecEnv([lambda: self.env])
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            policy=self.rl_model.policy,
            rng=self.rng,
        )
        # 测评模型
        print("正在评估 raw BC model ......")
        bc_raw_reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
        # 储存模型
        if not os.path.exists(self.bc_policy_path):
            os.makedirs(self.bc_policy_path)
        # 训练BC
        print("正在训练 BC model ......")
        bc_trainer.train(n_epochs=self.n_bc_train_epochs)
        print("正在储存训练完的 BC model ......")
        bc_trainer.save_policy(self.bc_policy_store_path)
        print("已储存至", self.bc_policy_store_path)
        # 测评模型
        print("正在评估训练完的 BC model ......")
        bc_trained_reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
        print("bc_raw_reward:", bc_raw_reward)
        print("bc_trained_reward:", bc_trained_reward)
        return

    def rl_train_after_pretrain(self):
        if self.env.method == pybullet.GUI:
            raise AssertionError("训练pybullet环境任务，请将env.method设置为pybullet.DIRECT")
        env = DummyVecEnv([lambda: self.env])
        rl_model = self.rl_model
        # 加载模型
        print("从", self.pretrained_model_path, "加载模型")
        bc_policy = bc.reconstruct_policy(self.pretrained_model_path)  # FeedForward32Policy
        # 测评模型
        load_bcpolicy_reward, _ = evaluate_policy(bc_policy, env, 10)
        print("load_bcpolicy_reward:", load_bcpolicy_reward)
        # policy migrate
        rl_model.policy = bc_policy
        pretrained_rlmodel_reward, _ = evaluate_policy(rl_model, env, 10)
        print("pretrained_rlmodel_reward:", pretrained_rlmodel_reward)
        # rl train
        eval_callback = EvalCallback(env,
                                     best_model_save_path=os.path.join(self.il_rl_path, 'eval'),
                                     log_path=os.path.join(self.il_rl_path, 'eval'),
                                     eval_freq=100,
                                     deterministic=True,
                                     render=False)
        rl_model.learn(self.n_rl_train_steps, callback=eval_callback)
        rl_model.save(self.il_rl_path)
        # 测评模型
        trained_rlmodel_reward, _ = evaluate_policy(rl_model, env, 10)
        # 打印结果
        print("load_bcpolicy_reward:", load_bcpolicy_reward)
        print("pretrained_rlmodel_reward:", pretrained_rlmodel_reward)
        print("trained_rlmodel_reward:", trained_rlmodel_reward)

    def test_bc_model(self, model_path, n_step=500):
        test_env = self.env
        bc_policy = bc.reconstruct_policy(model_path)
        print("正在评估 bc_policy ......")
        reward, _ = evaluate_policy(bc_policy, test_env, 10)
        print("bc_policy_10episodes_average_reward:", reward)

        if test_env.method != pybullet.GUI:
            raise AssertionError("pybullet仿真环境不可视，请将env.method设置为pybullet.GUI")
        obs = test_env.reset()
        episode_reward = 0
        for _ in range(n_step):
            action, _state = bc_policy.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward
            sleep(0.005)
            if done:
                print("episode_reward =", episode_reward)
                obs = test_env.reset()
                episode_reward = 0


if __name__ == "__main__":
    '''导入环境'''
    # ur = DiscreteActPuzzle(method=pybullet.GUI, mode="test")
    # ur = DiscreteActPuzzle()
    # ur = MatrixPuzzleObsUR5Sim()
    ur = FlattenMatrixObsUR5Sim()
    '''测试环境'''
    # check_env(ur)
    # print("check over !!!")

    '''操作环境'''
    trainer = OperateRobot(ur)

    '''rl train'''
    # trainer.rl_train("sac")
    # trainer.venv_rl_train("sac")
    '''rl retrain'''
    # trainer.rl_retrain(model_path='ur5e_puzzle_result/rl/MatrixObsEnv/23-06-22/15:04/eventual_model.zip',
    #                    store_path='ur5e_puzzle_result/rl/MatrixObsEnv/23-06-22/15:04',
    #                    rl_algorithm="ppo")

    '''test rl model'''
    # test_model_path = 'ur5e_puzzle_result/rl/DiscreteActPuzzle/23-06-21 00/00:40/eval/best_model.zip'
    # trainer.test_rl_model("ppo", test_model_path)

    '''il'''
    transitions = trainer.rollouts_to_transitions(
        'ur5e_maze_rollouts/FlattenMatrixObsEnv/rollouts100 23-06-26 22:06.pkl')
    trainer.bc_pretrain(transitions)

    '''rl after il'''
    # trainer.rl_train_after_pretrain()


